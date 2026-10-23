"""12GB GPU(RTX A2000)용: Stage C/B 생성기를 번갈아 GPU에 올린다. 스모크에서 모델을 전부 올리면 12.6GB(최대 13.9GB)라 12GB에는 안 들어간다.
이 패치는 실행 검증 전이다(코드를 읽고 설계). 원본은 .orig로 백업한다."""
import sys, shutil, os, ast
f = sys.argv[1]
if not os.path.exists(f + ".orig"): shutil.copy(f, f + ".orig")
s = open(f + ".orig").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, f"앵커를 찾지 못했거나 중복: {a[:60]!r}"
    s = s.replace(a, b)
rep('        print("STAGE C READY")\n',
    '        print("STAGE C READY")\n        self.models.generator.to("cpu"); torch.cuda.empty_cache()   # [offload] 생성기는 샘플링할 때만 GPU로\n')
rep('            print("STAGE B READY")\n',
    '            print("STAGE B READY")\n            self.models_b.generator.to("cpu"); self.models_b.stage_a.to("cpu"); torch.cuda.empty_cache()   # [offload]\n')
rep('    def embed_image(self, images, hidden=False):',
    '''    def _stage(self, which):   # [offload]
        c, b, a = self.models.generator, self.models_b.generator, self.models_b.stage_a
        if which == "C":
            b.to("cpu"); a.to("cpu"); c.to(self.device)
        else:
            c.to("cpu"); b.to(self.device); a.to(self.device)
        torch.cuda.empty_cache()

    def embed_image(self, images, hidden=False):''')
rep('            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):\n                # Sampling Stage C',
    '            self._stage("C")   # [offload]\n            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):\n                # Sampling Stage C')
rep('        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):\n            # Sampling Stage B',
    '        self._stage("B")   # [offload]\n        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):\n            # Sampling Stage B')
ast.parse(s)
open(f, "w").write(s)
print("offload 패치 적용:", f)
