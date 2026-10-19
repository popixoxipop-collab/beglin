// BEGLIN research-only native C++ checkpoint-boundary admission.
// Reads only pinned safetensors headers and tensor metadata. No weight payload,
// deserialization, quantization, live model or deployment mutation.
// Apple Silicon native CommonCrypto implementation; fixture tests are isolated.
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
constexpr const char* PIN_INDEX =
  "d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf";
constexpr const char* PIN_CONFIG =
  "f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d";
constexpr const char* PIN_CODE =
  "7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c";
constexpr const char* PIN_MANIFEST =
  "274247e626f67579e56256e356f554fd1b1e5e5967df2603ec82bfb902af4963";
constexpr uint64_t MAX_FILE = 20ULL * 1024 * 1024 * 1024;
constexpr uint64_t MAX_MANIFEST = 3ULL * 1024 * 1024;

void gate(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error("BEGLIN_CPP_BOUNDARY_FAIL_CLOSED: " + what);
}
uint64_t number(const std::string& s, uint64_t max=MAX_FILE) {
    gate(!s.empty() && s.size() <= 20, "invalid decimal length");
    for (char c:s) gate(c >= '0' && c <= '9', "nondecimal unsigned field");
    errno=0;
    char* end=nullptr;
    unsigned long long v=std::strtoull(s.c_str(),&end,10);
    gate(errno==0 && end && *end=='\0' && v<=max, "decimal overflow/cap");
    return static_cast<uint64_t>(v);
}
uint64_t mul(uint64_t a, uint64_t b) {
    gate(b==0 || a<=std::numeric_limits<uint64_t>::max()/b, "integer overflow");
    return a*b;
}
uint64_t add(uint64_t a, uint64_t b) {
    gate(a<=std::numeric_limits<uint64_t>::max()-b, "integer addition overflow");
    return a+b;
}
bool hex64(const std::string& s) {
    if(s.size()!=64)return false;
    for(char c:s)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;
    return true;
}
std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    size_t start=0;
    while(true) {
        auto p=s.find(delim,start);
        out.push_back(s.substr(start,p==std::string::npos ? p:p-start));
        if(p==std::string::npos)break;
        start=p+1;
    }
    return out;
}
std::string hex_sha(const unsigned char* data, size_t size) {
    gate(size<=std::numeric_limits<CC_LONG>::max(),"SHA input too large");
    unsigned char result[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data,(CC_LONG)size,result);
    static constexpr char chars[]="0123456789abcdef";
    std::string out;
    out.reserve(64);
    for(unsigned char x:result) {out.push_back(chars[x>>4]);out.push_back(chars[x&15]);}
    return out;
}
struct Fd {
    int fd=-1;
    explicit Fd(const std::string& path) {
        fd=::open(path.c_str(),O_RDONLY|O_NOFOLLOW|O_CLOEXEC);
        gate(fd>=0,"missing/untrusted source regular file");
    }
    Fd(const Fd&)=delete;
    Fd& operator=(const Fd&)=delete;
    ~Fd(){if(fd>=0)::close(fd);}
    uint64_t size()const {
        struct stat s {};
        gate(::fstat(fd,&s)==0 && S_ISREG(s.st_mode) && s.st_size>0
             && static_cast<uint64_t>(s.st_size)<=MAX_FILE,
             "source not a bounded regular file");
        return static_cast<uint64_t>(s.st_size);
    }
    std::vector<unsigned char> read(uint64_t pos,size_t count)const {
        gate(pos<=MAX_FILE && count<=MAX_MANIFEST && pos+count<=size(),
             "pread span out of actual file");
        std::vector<unsigned char> out(count);
        size_t done=0;
        while(done<count) {
            ssize_t n=::pread(fd,out.data()+done,count-done,static_cast<off_t>(pos+done));
            gate(n>0,"source pread short/error");
            done+=static_cast<size_t>(n);
        }
        return out;
    }
};
std::string sha_file(const std::string& path,size_t maxlen) {
    Fd f(path);
    const auto len=f.size();
    gate(len<=maxlen,"metadata source exceeds bound");
    auto bytes=f.read(0,static_cast<size_t>(len));
    return hex_sha(bytes.data(),bytes.size());
}
struct Shard {
    std::string name;
    uint64_t actual=0,header_len=0,body_len=0,declared_count=0;
    std::string header_sha;
    std::vector<std::pair<uint64_t,uint64_t>> segments;
};
struct Header {
    std::string index_sha;
    uint64_t n_all=0,n_quant=0,layers=0,n_shards=0,qbytes=0,bf16bytes=0;
};
uint64_t little_u64(const std::vector<unsigned char>& v){
    gate(v.size()==8,"bad u64 raw source");
    uint64_t out=0;
    for(unsigned i=0;i<8;i++) out|=(uint64_t)v[i]<<(i*8);
    return out;
}
bool legal_filename(const std::string& s) {
    static const std::regex pattern(R"(model-[0-9]{5}-of-[0-9]{6}[.]safetensors)");
    return std::regex_match(s,pattern);
}
bool legal_tensor(const std::string& s) {
    static const std::regex pattern(R"([A-Za-z0-9_.]+)");
    return std::regex_match(s,pattern);
}
bool expected_quant_role(const std::string& name,int& layer){
    static const std::regex r(
        R"(model[.]layers[.]([0-9]+)[.](self_attn[.](q_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)|mlp[.](gate_proj|up_proj|down_proj|shared_experts[.](gate_proj|up_proj|down_proj)|experts[.]([0-9]+)[.](gate_proj|up_proj|down_proj)))[.]weight)");
    std::smatch m;
    if(!std::regex_match(name,m,r))return false;
    const auto l=number(m[1].str(),26);
    layer=static_cast<int>(l);
    std::string role=m[2].str();
    bool moe=(role.find("mlp.shared_experts.")==0 || role.find("mlp.experts.")==0);
    if(l==0 && moe)return false;
    if(l>0 && role.find("mlp.")==0 && !moe)return false;
    if(m[6].matched) gate(number(m[6].str(),63)<64,"source expert id out of bound");
    return true;
}
std::string inspect(const std::string& mode,const std::string& dir,
                    const std::string& manifest_path) {
    const bool real=(mode=="--real");
    gate(real || mode=="--fixture","supported mode only");
    gate(dir.find('\0')==std::string::npos && !dir.empty(),"source path bad");
    gate(manifest_path.find('\0')==std::string::npos,"manifest path bad");
    if(real){
        char canonical[PATH_MAX];gate(::realpath(dir.c_str(),canonical)!=nullptr,"canonical root");
        gate(std::string(canonical)==dir,"source directory must be canonical");
        gate(sha_file(dir+"/model.safetensors.index.json",2'000'000)==PIN_INDEX,
             "pinned checkpoint index SHA");
        gate(sha_file(dir+"/config.json",2'000'000)==PIN_CONFIG,"pinned config SHA");
        gate(sha_file(dir+"/modeling_deepseek.py",2'000'000)==PIN_CODE,"pinned source SHA");
    }
    Fd manifest(manifest_path);
    uint64_t file_len=manifest.size();
    gate(file_len<=MAX_MANIFEST,"manifest too big");
    auto contents=manifest.read(0,(size_t)file_len);
    std::string manifest_sha=hex_sha(contents.data(),contents.size());
    if(real) gate(manifest_sha==PIN_MANIFEST,"unapproved manifest digest");
    gate(contents.back()=='\n',"manifest lacks final newline");
    std::string text(contents.begin(),contents.end());
    gate(text.find('\r')==std::string::npos && text.find('\0')==std::string::npos,
         "invalid text encoding or line terminator");
    std::istringstream input(text);
    std::string line;
    gate((bool)std::getline(input,line),"missing manifest header");
    auto v=split(line,'\t');
    gate(v.size()==8 && v[0]=="BEGLIN-SOURCE-MANIFEST/1" && hex64(v[1]),
         "manifest header schema");
    Header h{v[1],number(v[2],10000),number(v[3],10000),number(v[4],64),
        number(v[5],16),number(v[6],MAX_FILE),number(v[7],MAX_FILE*2)};
    if(real)gate(h.index_sha==PIN_INDEX && h.n_all==5291 && h.n_quant==5181 &&
        h.layers==27 && h.n_shards==4 && h.qbytes==14328299520ULL &&
        h.bf16bytes==31412968448ULL,"full27 fixed source size/count contract");
    std::map<std::string,Shard> shards;
    std::set<std::string> keys;
    uint64_t n_all=0,n_quant=0,qbytes=0,bf16=0;
    std::vector<uint64_t> layer_counts(h.layers,0);
    size_t line_no=1;
    while(std::getline(input,line)) {
        line_no++;
        gate(line_no<=20000 && !line.empty(),"excess/empty manifest line");
        auto row=split(line,'\t');
        gate(row.size()>=2,"manifest row width");
        if(row[0]=="S") {
            gate(row.size()==6 && legal_filename(row[1]) && hex64(row[4]),
                 "shard row schema");
            gate(shards.count(row[1])==0 && shards.size()<h.n_shards,"duplicate/unlisted shard");
            Shard s;
            s.name=row[1];
            s.actual=number(row[2],MAX_FILE);
            s.header_len=number(row[3],MAX_MANIFEST);
            s.header_sha=row[4];
            s.declared_count=number(row[5],10000);
            gate(s.header_len>0 && s.declared_count>0,"invalid shard metadata");
            Fd f(dir+"/"+s.name);
            const auto bytes=f.size();
            gate(bytes==s.actual && bytes>=s.header_len+8,"shard size mismatch");
            gate(little_u64(f.read(0,8))==s.header_len,"actual header length mismatch");
            const auto header=f.read(8,(size_t)s.header_len);
            gate(hex_sha(header.data(),header.size())==s.header_sha,
                 "actual safetensors header SHA drift");
            s.body_len=bytes-8-s.header_len;
            shards.emplace(s.name,std::move(s));
        }else if(row[0]=="T"){
            gate(row.size()==9 && legal_tensor(row[1]) && row[3]=="BF16",
                 "invalid tensor name/row schema/dtype");
            gate(keys.insert(row[1]).second,"duplicate tensor name");
            auto sh=shards.find(row[2]);
            gate(sh!=shards.end(),"tensor references missing/late shard");
            auto dims=split(row[4],',');
            gate(dims.size()==1 || dims.size()==2,"BF16 tensor rank violation");
            uint64_t elems=1;
            for(const auto& d:dims){
                auto dimension=number(d,1000000);
                gate(dimension>0,"empty tensor dimension");
                elems=mul(elems,dimension);
            }
            auto begin=number(row[5],MAX_FILE);
            auto end=number(row[6],MAX_FILE);
            gate(begin<end && end<=sh->second.body_len &&
                 end-begin==mul(elems,2),"tensor out-of-bounds/size mismatch");
            const auto bit=number(row[7],16);
            const auto packed=number(row[8],MAX_FILE);
            int layer=0;
            const bool quantizable=expected_quant_role(row[1],layer);
            gate((bit==7)==quantizable && (bit==7 || bit==0),
                 "quantization role/bit invariant");
            if(bit==7){
                gate(dims.size()==2 && number(dims[1])%64==0 && elems%64==0,
                     "group64 dimension missing");
                const auto wanted=add(mul(elems,7)/8,mul(elems/64,4));
                gate(packed==wanted,"encoded bitplane and F32 scale byte budget mismatch");
                gate(static_cast<size_t>(layer)<layer_counts.size(),
                     "model layer limit");
                layer_counts[layer]++;
                n_quant++;
                qbytes=add(qbytes,packed);
            } else gate(packed==0,"nonquantized roles must have zero packed bytes");
            n_all++;
            bf16=add(bf16,mul(elems,2));
            sh->second.segments.emplace_back(begin,end);
        }else gate(false,"unknown manifest row type");
    }
    gate(shards.size()==h.n_shards && n_all==h.n_all &&
        n_quant==h.n_quant && qbytes==h.qbytes &&
        bf16==h.bf16bytes,"header totals differ from parsed tensors");
    uint64_t payload_sum=0;
    for(auto& it:shards){
        auto& s=it.second;
        gate(s.segments.size()==s.declared_count,
             "per-shard tensor count differs from header");
        std::sort(s.segments.begin(),s.segments.end());
        uint64_t cursor=0;
        for(auto pair:s.segments){
            gate(pair.first==cursor && pair.second>pair.first,
                 "source tensor gap, overlap, or reordered range");
            cursor=pair.second;
        }
        gate(cursor==s.body_len,"unclaimed / overclaimed shard payload bytes");
        payload_sum=add(payload_sum,cursor);
    }
    gate(payload_sum==h.bf16bytes,"all shard BF16 bytes do not sum");
    if(real) {
        gate(layer_counts.size()==27,"invalid 27-layer count");
        for(size_t i=0;i<27;i++)
            gate(layer_counts[i]==(i==0 ? 7U:199U),"incorrect per-layer tensor coverage");
    }
    std::ostringstream output;
    output<<"{\"status\":\"PASS_CPP_SOURCE_MANIFEST_BOUNDS_NO_WEIGHT_PAYLOAD\""
          <<",\"real_source\":"<<(real?"true":"false")
          <<",\"shards\":"<<shards.size()
          <<",\"tensors\":"<<n_all
          <<",\"qng64_matrices\":"<<n_quant
          <<",\"layers\":"<<h.layers
          <<",\"source_BF16_bytes\":"<<bf16
          <<",\"serialized_qng64_bytes\":"<<qbytes
          <<",\"source_manifest_sha256\":\""<<manifest_sha<<"\"}";
    return output.str();
}
}
int main(int argc,char** argv){
    try{
        gate(argc==4,"usage: <program> --real|--fixture source_directory manifest.tsv");
        std::cout<<inspect(argv[1],argv[2],argv[3])<<"\n";
        return 0;
    }catch(const std::exception& e){
        std::cerr<<e.what()<<"\n";
        return 2;
    }
}
