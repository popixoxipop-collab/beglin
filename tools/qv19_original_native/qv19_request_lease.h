#ifndef BEGLIN_QV19_REQUEST_LEASE_H
#define BEGLIN_QV19_REQUEST_LEASE_H

#include <pthread.h>
#include <stdint.h>

/* Research-level interface. Every read/write of mutable model weights, KV cache,
 * activations and logits must happen INSIDE a qv19_run_request callback. */
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    QV19_OK = 0,
    QV19_BUSY = 1,
    QV19_ERR_INPUT = -1,
    QV19_ERR_MUTEX = -2,
    QV19_ERR_POISONED = -3,
    QV19_ERR_PRECISION = -4,
    QV19_ERR_FORWARD = -5,
    QV19_ERR_INVARIANT = -6
} qv19_result;

typedef int (*qv19_apply_precision_fn)(void *context, int target_profile);
typedef int (*qv19_forward_and_score_fn)(void *context, double *gold_nll);

typedef struct {
    pthread_mutex_t exclusive;
    uint64_t epoch;
    uint64_t completed;
    uint64_t precision_changes;
    uint32_t active;
    uint32_t peak_active;
    int current_profile;
    int poisoned;
    int initialized;
} qv19_model_lease;

typedef struct {
    uint64_t epoch_enter;
    uint64_t epoch_exit;
    uint64_t lock_wait_ns;
    uint64_t guarded_duration_ns;
    int selected_profile;
    int profile_changed;
    double gold_nll;
} qv19_request_receipt;

typedef struct {
    uint64_t epoch;
    uint64_t completed;
    uint64_t precision_changes;
    uint32_t active;
    uint32_t peak_active;
    int current_profile;
    int poisoned;
} qv19_lease_snapshot;

qv19_result qv19_model_lease_init(qv19_model_lease *model, int initial_profile);
/* The caller must have stopped and joined all request threads before destroy. */
qv19_result qv19_model_lease_destroy(qv19_model_lease *model);

/* Serializes precision switch, full forward including shared KV, logits and
 * output calculation. Calls must not escape the lease with retained WT pointers.
 * Any failed callback poisons the lease (no unverified partial state reuse).
 * try_only=1 uses pthread_mutex_trylock and returns QV19_BUSY rather than waiting. */
qv19_result qv19_run_request(qv19_model_lease *model, int target_profile,
                             qv19_apply_precision_fn apply_precision,
                             qv19_forward_and_score_fn forward_and_score,
                             void *context, int try_only,
                             qv19_request_receipt *receipt);
qv19_result qv19_lease_get_snapshot(qv19_model_lease *model,
                                    qv19_lease_snapshot *snapshot);

#ifdef __cplusplus
}
#endif
#endif
