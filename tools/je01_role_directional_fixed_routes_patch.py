#!/usr/bin/env python3
"""Optional, separate fixed-hard-route derivative diagnostic for JE01.

Apply to the exact already-verified role-direction source, never to a running
build. No replay environment variable means the ordinary routing arrays are
returned unchanged. Replay substitutes only hard Expert indices; current
softmax scores and the selected-weight renormalization remain dynamic.
Alpha one is forbidden with replay because physical endpoints use ordinary
native routing. This is a branch derivative diagnostic, not a serving policy.

The runner must verify the original OFF trace SHA/size/record count and bind it
to the same model, dataset, case order, metadata and baseline loss receipt.
Native code additionally checks every replay event coordinate and exact EOF.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

CPP_SHA256 = "619b88780373a8d80d5e15e9236358af39a363458608bbc26e6006bdb677f62b"
C_SHA256 = "8d1a8fce7d578319ceb45d17948ebe0349758acb2ae0b786944c8119ea795c3d"
ROUTE_MAGIC = b"JE01_ROLE_DIRECTION_ROUTES_V1\n"
MARGIN_MAGIC = b"JE01_ROLE_DIRECTION_BASELINE_MARGINS_V1\n"
MAX_ROUTE_BYTES = 32*1024*1024

HELPERS = r'''
// Optional hard-branch replay. Scores/weights are still recomputed normally.
static bool g_je01_fixed_routes_initialized=false;
static std::ifstream g_je01_fixed_routes;
static uint32_t g_je01_fixed_route_event=0;
static uint64_t g_je01_fixed_route_expected=0;
static bool g_je01_fixed_routes_active=false;

static bool je01_direction_optional_path(const char *name) {
    const char *p=std::getenv(name);return p && *p;
}

static mx::array je01_direction_fixed_routes(int layer,int A,const int *slot,
    const int *spos,const mx::array &computed_indices) {
    if(!g_je01_fixed_routes_initialized) {
        g_je01_fixed_routes_initialized=true;
        const char *path=std::getenv("JE01_DIRECTION_BASELINE_ROUTES");
        if(path && *path) {
            if(g_je01_direction_alpha==1.0f)
                throw std::runtime_error("JE01_FIXED_ROUTES_ALPHA1_ENDPOINT_FORBIDDEN");
            const std::string prefix="/Volumes/D50/TailnetCommander/work/je01_";
            namespace fs=std::filesystem;
            fs::path p(path);
            if(std::string(path).compare(0,prefix.size(),prefix)!=0 ||
               fs::is_symlink(p) || !fs::is_regular_file(p))
                throw std::runtime_error("JE01_FIXED_ROUTES_SOURCE_PATH_INVALID");
            const char magic[]="JE01_ROLE_DIRECTION_ROUTES_V1\n";
            const uint64_t n=fs::file_size(p),header=sizeof(magic)-1;
            if(n<=header || n>33554432 || (n-header)%40!=0 || ((n-header)/40)%26!=0)
                throw std::runtime_error("JE01_FIXED_ROUTES_SOURCE_SIZE_INVALID");
            g_je01_fixed_route_expected=(n-header)/40;
            g_je01_fixed_routes.open(path,std::ios::binary);
            char found[sizeof(magic)-1];
            if(!g_je01_fixed_routes.read(found,sizeof(found)) ||
               std::memcmp(found,magic,sizeof(found))!=0)
                throw std::runtime_error("JE01_FIXED_ROUTES_SOURCE_MAGIC_INVALID");
            g_je01_fixed_routes_active=true;
            std::fprintf(stderr,"JE01_FIXED_BASELINE_ROUTES_ENABLED records=%llu\n",
                         (unsigned long long)g_je01_fixed_route_expected);
        }
    }
    if(!g_je01_fixed_routes_active)return computed_indices;
    if(A!=1 || !slot || !spos || slot[0]!=0 || spos[0]<0 || spos[0]>=384 ||
       computed_indices.ndim()!=2 || computed_indices.shape(0)!=1 ||
       computed_indices.shape(1)!=6 || computed_indices.dtype()!=mx::int32 ||
       g_je01_fixed_route_event>=g_je01_fixed_route_expected)
        throw std::runtime_error("JE01_FIXED_ROUTES_CALL_OR_LENGTH_MISMATCH");
    int32_t record[10];
    if(!g_je01_fixed_routes.read((char*)record,sizeof(record)))
        throw std::runtime_error("JE01_FIXED_ROUTES_TRUNCATED");
    if(record[0]!=(int32_t)g_je01_fixed_route_event || record[1]!=layer ||
       record[2]!=slot[0] || record[3]!=spos[0] ||
       layer!=(int)(g_je01_fixed_route_event%26)+1)
        throw std::runtime_error("JE01_FIXED_ROUTES_EVENT_ALIGNMENT");
    for(int i=4;i<10;i++) {
        if(record[i]<0 || record[i]>=64)
            throw std::runtime_error("JE01_FIXED_ROUTES_EXPERT_RANGE");
        for(int j=4;j<i;j++)if(record[i]==record[j])
            throw std::runtime_error("JE01_FIXED_ROUTES_DUPLICATE_EXPERT");
    }
    ++g_je01_fixed_route_event;
    // Three-argument iterator constructor copies stack-backed IDs immediately.
    return mx::array(record+4,{1,6},mx::int32);
}

static int je01_direction_fixed_routes_finish(void) {
    if(!g_je01_fixed_routes_active)return 1;
    int ok=g_je01_fixed_route_event==g_je01_fixed_route_expected &&
           g_je01_fixed_routes.peek()==std::char_traits<char>::eof();
    // EOF is expected, so clear that flag before testing the close itself.
    g_je01_fixed_routes.clear();
    g_je01_fixed_routes.close();
    if(g_je01_fixed_routes.fail())ok=0;
    return ok;
}

// Optional margins on the original alpha-zero branch. No extra eval boundary.
struct JE01BaselineMarginPending {
    int32_t event,layer,slot,position;
    mx::array scores,indices;
};
static FILE *g_je01_baseline_margin_file=nullptr;
static bool g_je01_baseline_margin_initialized=false;
static uint32_t g_je01_baseline_margin_event=0;
static std::vector<JE01BaselineMarginPending> g_je01_baseline_margin_pending;

static void je01_direction_baseline_margin_capture(int layer,int A,const int *slot,
    const int *spos,const mx::array &scores,const mx::array &computed_indices) {
    if(!g_je01_baseline_margin_initialized) {
        g_je01_baseline_margin_initialized=true;
        const char *path=std::getenv("JE01_DIRECTION_MARGIN_OUTPUT");
        if(path && *path) {
            if(g_je01_direction_alpha!=0.0f)
                throw std::runtime_error("JE01_BASELINE_MARGIN_REQUIRES_ALPHA_ZERO");
            const std::string prefix="/Volumes/D50/TailnetCommander/work/je01_";
            if(std::string(path).compare(0,prefix.size(),prefix)!=0)
                throw std::runtime_error("JE01_BASELINE_MARGIN_RESEARCH_PATH_REQUIRED");
            int fd=::open(path,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
            if(fd<0)throw std::runtime_error("JE01_BASELINE_MARGIN_MUST_BE_NEW");
            g_je01_baseline_margin_file=::fdopen(fd,"wb");
            if(!g_je01_baseline_margin_file) {::close(fd);
                throw std::runtime_error("JE01_BASELINE_MARGIN_OPEN_FAILED");}
            const char magic[]="JE01_ROLE_DIRECTION_BASELINE_MARGINS_V1\n";
            if(std::fwrite(magic,1,sizeof(magic)-1,g_je01_baseline_margin_file)!=sizeof(magic)-1)
                throw std::runtime_error("JE01_BASELINE_MARGIN_HEADER_WRITE_FAILED");
        }
    }
    if(!g_je01_baseline_margin_file)return;
    if(A!=1 || !slot || !spos || slot[0]!=0 || spos[0]<0 || spos[0]>=384 ||
       scores.ndim()!=2 || scores.shape(0)!=1 || scores.shape(1)!=64 || scores.dtype()!=mx::float32 ||
       computed_indices.ndim()!=2 || computed_indices.shape(0)!=1 ||
       computed_indices.shape(1)!=6 || computed_indices.dtype()!=mx::int32 ||
       g_je01_baseline_margin_event>=10000000 || g_je01_baseline_margin_pending.size()>=26)
        throw std::runtime_error("JE01_BASELINE_MARGIN_SHAPE_OR_LIMIT");
    g_je01_baseline_margin_pending.push_back({(int32_t)g_je01_baseline_margin_event++,
                                              layer,slot[0],spos[0],scores,computed_indices});
}

static void je01_direction_baseline_margin_append(std::vector<mx::array> &outputs) {
    for(const auto &r:g_je01_baseline_margin_pending) {
        outputs.push_back(r.scores);outputs.push_back(r.indices);
    }
}

static void je01_direction_baseline_margin_emit(void) {
    if(!g_je01_baseline_margin_file) {
        if(!g_je01_baseline_margin_pending.empty())
            throw std::runtime_error("JE01_BASELINE_MARGIN_PENDING_WITHOUT_FILE");
        return;
    }
    if(g_je01_baseline_margin_pending.size()!=26)
        throw std::runtime_error("JE01_BASELINE_MARGIN_NOT_ALL26_LAYERS");
    for(size_t row=0;row<g_je01_baseline_margin_pending.size();row++) {
        const auto &r=g_je01_baseline_margin_pending[row];
        if(r.layer!=(int)row+1)throw std::runtime_error("JE01_BASELINE_MARGIN_LAYER_ORDER");
        const float *scores=r.scores.data<float>();
        const int32_t *ids=r.indices.data<int32_t>();
        bool selected[64]={false};
        for(int i=0;i<6;i++) {
            if(ids[i]<0 || ids[i]>=64 || selected[ids[i]])
                throw std::runtime_error("JE01_BASELINE_MARGIN_EXPERT_IDS_INVALID");
            selected[ids[i]]=true;
        }
        float minimum_selected=INFINITY,maximum_unselected=-INFINITY;
        for(int i=0;i<64;i++) {
            float p=scores[i];
            if(!std::isfinite(p) || p<0.0f || p>1.0f)
                throw std::runtime_error("JE01_BASELINE_MARGIN_NONFINITE_PROBABILITY");
            if(selected[i])minimum_selected=std::min(minimum_selected,p);
            else maximum_unselected=std::max(maximum_unselected,p);
        }
        float values[3]={minimum_selected,maximum_unselected,minimum_selected-maximum_unselected};
        if(!std::isfinite(values[2]) || values[2]<0.0f)
            throw std::runtime_error("JE01_BASELINE_MARGIN_NOT_TOPK");
        int32_t coordinates[4]={r.event,r.layer,r.slot,r.position};
        if(std::fwrite(coordinates,sizeof(int32_t),4,g_je01_baseline_margin_file)!=4 ||
           std::fwrite(values,sizeof(float),3,g_je01_baseline_margin_file)!=3)
            throw std::runtime_error("JE01_BASELINE_MARGIN_WRITE_FAILED");
    }
    g_je01_baseline_margin_pending.clear();
}

static int je01_direction_baseline_margin_finish(void) {
    int ok=g_je01_baseline_margin_pending.empty();
    if(!g_je01_baseline_margin_file)return ok;
    FILE *f=g_je01_baseline_margin_file;g_je01_baseline_margin_file=nullptr;
    if(std::fflush(f)!=0)ok=0;
    if(::fsync(::fileno(f))!=0)ok=0;
    if(std::fclose(f)!=0)ok=0;
    return ok;
}
'''

FINISH = r'''
extern "C" int mlx_gpu_je01_direction_trace_finish(void) {
    int original_ok=je01_direction_original_trace_finish();
    int replay_ok=je01_direction_fixed_routes_finish();
    int margin_ok=je01_direction_baseline_margin_finish();
    return original_ok && replay_ok && margin_ok;
}
'''


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def once(text, old, new, label):
    if text.count(old) != 1:
        raise ValueError("JE01_FIXED_ROUTE_ANCHOR_"+label+"_"+str(text.count(old)))
    return text.replace(old, new, 1)


def apply(cpp, c):
    if digest(cpp) != CPP_SHA256 or digest(c) != C_SHA256:
        raise ValueError("JE01_FIXED_ROUTES_BASE_DIRECTION_SOURCE_SHA_DRIFT")
    anchor = "static void je01_direction_route_trace("
    cpp = once(cpp, anchor, HELPERS+"\n"+anchor, "HELPERS")
    cpp = once(cpp, 'extern "C" int mlx_gpu_je01_direction_trace_finish(void) {',
               "static int je01_direction_original_trace_finish(void) {", "ORIGINAL_CLOSE")
    role = "static mx::array je01_direction_role_output("
    cpp = once(cpp, role, FINISH+"\n"+role, "COMBINED_CLOSE")
    start = cpp.index("int mlx_gpu_cbatch_layer_step_lazy(")
    stop = cpp.index("int mlx_gpu_cbatch_forward_finalize(", start)
    block = cpp[start:stop]
    selection = "            mx::array top_idx = mx::astype(top_idx_u, mx::int32);"
    block = once(block, selection, selection+
                 "\n            je01_direction_baseline_margin_capture(l,A,slot,spos,scores,top_idx);"+
                 "\n            top_idx=je01_direction_fixed_routes(l,A,slot,spos,top_idx);",
                 "BEFORE_DYNAMIC_SELECTED_WEIGHTS")
    cpp = cpp[:start]+block+cpp[stop:]
    start = cpp.index("int mlx_gpu_cbatch_forward_finalize(")
    stop = cpp.index("\n}\n", start)+3
    block = cpp[start:stop]
    block = once(block, "        je01_direction_route_append(all);",
                 "        je01_direction_route_append(all);\n        je01_direction_baseline_margin_append(all);",
                 "DEFERRED_MARGIN_APPEND")
    block = once(block, "        je01_direction_route_emit();",
                 "        je01_direction_route_emit();\n        je01_direction_baseline_margin_emit();",
                 "DEFERRED_MARGIN_EMIT")
    cpp = cpp[:start]+block+cpp[stop:]
    # Reject replay with alpha1 before the research bank is read or bound.
    parse_end = '        else return 0;\n        namespace fs=std::filesystem;'
    cpp = once(cpp, parse_end, '        else return 0;\n'+
               '        if(alpha==1.0f && je01_direction_optional_path("JE01_DIRECTION_BASELINE_ROUTES"))return 0;\n'+
               '        namespace fs=std::filesystem;', "NO_FIXED_ENDPOINT")
    return cpp, c


def parse_baseline_routes(raw):
    """Independent reader used to preflight a candidate original-OFF trace."""
    if not raw.startswith(ROUTE_MAGIC) or len(raw)>MAX_ROUTE_BYTES:
        raise ValueError("JE01_FIXED_ROUTES_MAGIC_OR_SIZE")
    body = raw[len(ROUTE_MAGIC):]
    if not body or len(body)%40 or (len(body)//40)%26:
        raise ValueError("JE01_FIXED_ROUTES_RECORD_EXTENT")
    records = list(struct.iter_unpack("<10i", body))
    previous_position = None
    for i, record in enumerate(records):
        event, layer, slot, position, *ids = record
        if event != i or layer != i%26+1 or slot != 0 or not 0<=position<384:
            raise ValueError("JE01_FIXED_ROUTES_EVENT_ALIGNMENT")
        if len(set(ids)) != 6 or not all(0<=x<64 for x in ids):
            raise ValueError("JE01_FIXED_ROUTES_INVALID_EXPERT_IDS")
        if i%26:
            if position != records[i-1][3]:
                raise ValueError("JE01_FIXED_ROUTES_LAYER_POSITION_MISMATCH")
        else:
            if previous_position is None and position != 0:
                raise ValueError("JE01_FIXED_ROUTES_SEQUENCE_START")
            if previous_position is not None and position not in (0, previous_position+1):
                raise ValueError("JE01_FIXED_ROUTES_SEQUENCE_CONTINUITY")
            previous_position = position
    return records


class BaselineRouteReplay:
    """Reference alignment/consumption check, independent of MLX execution."""
    def __init__(self, raw):
        self.records = parse_baseline_routes(raw)
        self.cursor = 0

    def take(self, layer, slot, position):
        if self.cursor >= len(self.records):
            raise ValueError("JE01_FIXED_ROUTES_EXHAUSTED")
        record = self.records[self.cursor]
        if record[:4] != (self.cursor, layer, slot, position):
            raise ValueError("JE01_FIXED_ROUTES_EVENT_ALIGNMENT")
        self.cursor += 1
        return record[4:]

    def finish(self):
        if self.cursor != len(self.records):
            raise ValueError("JE01_FIXED_ROUTES_NOT_FULLY_CONSUMED")


def stage(source):
    source = Path(source)
    work = Path("/Volumes/D50/TailnetCommander/work")
    if source.is_symlink() or not source.is_dir() or source.name != "source" or \
       not source.resolve().is_relative_to(work) or "je01_" not in source.parent.name or \
       "fixed" not in source.parent.name:
        raise ValueError("JE01_FIXED_ROUTES_SEPARATE_D50_SOURCE_REQUIRED")
    cpp, c = source/"mlx_moe.cpp", source/"qwen_infer.c"
    if any(not p.is_file() or p.is_symlink() for p in (cpp, c)):
        raise ValueError("JE01_FIXED_ROUTES_SOURCE_FILES_MISSING")
    new_cpp, new_c = apply(cpp.read_text(), c.read_text())
    cpp.write_text(new_cpp)
    if c.read_text() != new_c:
        raise ValueError("JE01_FIXED_ROUTES_C_SOURCE_UNEXPECTED_CHANGE")
    return {"schema": "je01-native-fixed-hard-route-diagnostic-source/1",
            "status": "SEPARATE_SCRATCH_SOURCE_PATCHED_NOT_EXECUTED",
            "base_cpp_sha256": CPP_SHA256, "unchanged_c_sha256": C_SHA256,
            "patched_cpp_sha256": digest(new_cpp),
            "replay_environment": "JE01_DIRECTION_BASELINE_ROUTES",
            "margin_environment": "JE01_DIRECTION_MARGIN_OUTPUT",
            "margin_magic": MARGIN_MAGIC.decode(),
            "margin_record": "little-endian <4i3f: event,layer,slot,pos,min_selected,max_unselected,gap",
            "current_softmax_and_selected_weights_dynamic": True,
            "alpha1_with_replay": "FORBIDDEN",
            "ordinary_no_environment_parity": "NOT_RUN",
            "native_derivative": "NOT_MEASURED", "production_modified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    print(json.dumps(stage(parser.parse_args().source), sort_keys=True))
