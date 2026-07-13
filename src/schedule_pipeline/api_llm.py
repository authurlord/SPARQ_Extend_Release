"""Drop-in shim that mimics vLLM offline LLM.generate() but routes to an OpenAI
/v1/completions endpoint (e.g. the shared 35B on 12.43). Prompts are sent raw
(already chat-templated by the caller) -> faithful replication of llm.generate."""
import os, concurrent.futures as cf, requests
for _k in ("http_proxy","https_proxy","all_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"):
    os.environ.pop(_k, None)

class _Out:
    __slots__=("text",)
    def __init__(self, text): self.text=text
class _Req:
    __slots__=("outputs",)
    def __init__(self, outs): self.outputs=outs

class ApiLLM:
    def __init__(self, api_base, model, api_key="EMPTY", concurrency=24, timeout=240):
        self.base=api_base.rstrip("/"); self.model=model; self.key=api_key
        self.conc=concurrency; self.timeout=timeout
    def _one(self, prompt, sp):
        body={"model":self.model,"prompt":prompt,
              "max_tokens":int(getattr(sp,"max_tokens",2048)),
              "temperature":float(getattr(sp,"temperature",0.0)),
              "top_p":float(getattr(sp,"top_p",1.0)),
              "n":int(getattr(sp,"n",1) or 1),
              "presence_penalty":float(getattr(sp,"presence_penalty",0.0)),
              "chat_template_kwargs":{"enable_thinking":False}}
        tk=getattr(sp,"top_k",-1); mp=getattr(sp,"min_p",0.0)
        if tk and tk>0: body["top_k"]=int(tk)
        if mp and mp>0: body["min_p"]=float(mp)
        last=None
        for _ in range(4):
            try:
                r=requests.post(f"{self.base}/completions", json=body,
                                headers={"Authorization":f"Bearer {self.key}"}, timeout=self.timeout)
                r.raise_for_status()
                ch=sorted(r.json()["choices"], key=lambda c:c.get("index",0))
                return _Req([_Out(c.get("text","") or "") for c in ch])
            except Exception as e:
                last=e
        return _Req([_Out("")])
    def generate(self, prompts, sampling_params, **kw):
        if isinstance(prompts, str): prompts=[prompts]
        res=[None]*len(prompts)
        with cf.ThreadPoolExecutor(max_workers=self.conc) as ex:
            futs={ex.submit(self._one,p,sampling_params):i for i,p in enumerate(prompts)}
            for f in cf.as_completed(futs):
                res[futs[f]]=f.result()
        return res
