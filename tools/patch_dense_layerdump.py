#!/usr/bin/env python3
from pathlib import Path
p=Path("qwen_infer.c")
s=p.read_text()
old="        vDSP_vadd(xbuf,1,mlpout,1,xbuf,1,g_cfg.d);\n"
new="""        vDSP_vadd(xbuf,1,mlp_out,1,xbuf,1,g_cfg.d);
        {
            const char *dp=getenv("QWEN_DEBUG_LAYERDUMP");
            int target=0;
            const char *pe=getenv("QWEN_DEBUG_LAYERDUMP_POS");
            if(pe&&pe[0]) target=atoi(pe);
            if(dp&&dp[0]&&pos==target){
                static FILE *df=NULL;
                if(l==0) df=fopen(dp,"wb");
                if(!df){perror(dp);exit(1);}
                if(fwrite(xbuf,sizeof(float),g_cfg.d,df)!=(size_t)g_cfg.d){
                    fprintf(stderr,"FATAL: dense layerdump write failed\\n");exit(1);
                }
                if(l==g_cfg.nl-1){fclose(df);df=NULL;}
            }
        }
"""
if s.count(old)!=1:
    raise SystemExit("dense layerdump anchor mismatch")
p.write_text(s.replace(old,new,1))
