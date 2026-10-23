#define _POSIX_C_SOURCE 200809L
#include "qv19_request_lease.h"
#include <errno.h>
#include <math.h>
#include <string.h>
#include <time.h>

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

qv19_result qv19_model_lease_init(qv19_model_lease *model, int initial_profile) {
    if (!model || (initial_profile != 0 && initial_profile != 1)) return QV19_ERR_INPUT;
    memset(model, 0, sizeof *model);
    if (pthread_mutex_init(&model->exclusive, NULL) != 0) return QV19_ERR_MUTEX;
    model->current_profile = initial_profile;
    model->initialized = 1;
    return QV19_OK;
}

qv19_result qv19_model_lease_destroy(qv19_model_lease *model) {
    if (!model || !model->initialized) return QV19_ERR_INPUT;
    /* The application must join all workers first. EBUSY also fails closed. */
    if (pthread_mutex_destroy(&model->exclusive) != 0) return QV19_ERR_MUTEX;
    model->initialized = 0;
    return QV19_OK;
}

qv19_result qv19_run_request(qv19_model_lease *model, int target_profile,
                             qv19_apply_precision_fn apply_precision,
                             qv19_forward_and_score_fn forward_and_score,
                             void *context, int try_only,
                             qv19_request_receipt *receipt) {
    if (!model || !model->initialized || !apply_precision || !forward_and_score ||
        !receipt || (target_profile != 0 && target_profile != 1) ||
        (try_only != 0 && try_only != 1)) return QV19_ERR_INPUT;
    memset(receipt, 0, sizeof *receipt);

    uint64_t t_wait = monotonic_ns();
    int acquired = try_only ? pthread_mutex_trylock(&model->exclusive)
                            : pthread_mutex_lock(&model->exclusive);
    if (acquired == EBUSY && try_only) return QV19_BUSY;
    if (acquired != 0) return QV19_ERR_MUTEX;
    uint64_t entered = monotonic_ns();
    receipt->lock_wait_ns = entered >= t_wait ? entered - t_wait : 0;
    qv19_result code = QV19_OK;

    if (model->poisoned) { code = QV19_ERR_POISONED; goto finish; }
    if (model->active != 0 || model->current_profile < 0) {
        model->poisoned = 1; code = QV19_ERR_INVARIANT; goto finish;
    }
    model->active = 1;
    if (model->peak_active < model->active) model->peak_active = model->active;
    receipt->epoch_enter = ++model->epoch;
    receipt->selected_profile = target_profile;

    if (model->current_profile != target_profile) {
        /* Complete conversion/repacking must finish before entering forward. */
        if (apply_precision(context, target_profile) != 0) {
            model->current_profile = -1;
            model->poisoned = 1;
            code = QV19_ERR_PRECISION;
            goto leave_active;
        }
        model->current_profile = target_profile;
        model->precision_changes++;
        receipt->profile_changed = 1;
    }

    /* Full token forward, attention KV writes, final logits and NLL must be
     * synchronous and finish here. Asynchronous GPU jobs need device barriers. */
    if (forward_and_score(context, &receipt->gold_nll) != 0 ||
        !isfinite(receipt->gold_nll) || receipt->gold_nll < 0) {
        model->poisoned = 1;
        code = QV19_ERR_FORWARD;
        goto leave_active;
    }
    if (model->epoch != receipt->epoch_enter ||
        model->current_profile != target_profile || model->active != 1) {
        model->poisoned = 1;
        code = QV19_ERR_INVARIANT;
        goto leave_active;
    }
    receipt->epoch_exit = model->epoch;
    model->completed++;

leave_active:
    model->active = 0;
finish:
    {
        uint64_t exit_ns = monotonic_ns();
        receipt->guarded_duration_ns = exit_ns >= entered ? exit_ns - entered : 0;
        if (pthread_mutex_unlock(&model->exclusive) != 0) return QV19_ERR_MUTEX;
    }
    return code;
}

qv19_result qv19_lease_get_snapshot(qv19_model_lease *model,
                                    qv19_lease_snapshot *snapshot) {
    if (!model || !model->initialized || !snapshot) return QV19_ERR_INPUT;
    if (pthread_mutex_lock(&model->exclusive) != 0) return QV19_ERR_MUTEX;
    snapshot->epoch = model->epoch;
    snapshot->completed = model->completed;
    snapshot->precision_changes = model->precision_changes;
    snapshot->active = model->active;
    snapshot->peak_active = model->peak_active;
    snapshot->current_profile = model->current_profile;
    snapshot->poisoned = model->poisoned;
    if (pthread_mutex_unlock(&model->exclusive) != 0) return QV19_ERR_MUTEX;
    return QV19_OK;
}
