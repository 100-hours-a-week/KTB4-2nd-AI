"""검색 의존성의 호출 시간을 단계별로 기록하는 감싸기 객체

검색 알고리즘(app.search)은 그대로 두고, 구성 단계에서 의존성만 감싼다.
단계 이름은 interpret(질의 해석 VLM), embed(질의 임베딩), retrieve(후보 범위 벡터 검색),
sign_url(답변용 사진 URL 발급), answer(답변 VLM)이다. 같은 단계를 여러 번 부르면 합산된다.
"""

from app.core.tracing import current_trace, stage


class TracedVlm:
    def __init__(self, inner):
        self.inner = inner

    async def chat(
        self, messages, image_urls, max_tokens, response_format=None, *, temperature=None
    ):
        trace = current_trace()
        name = "interpret" if trace and trace.operation == "parse" else "answer"
        # 값이 있을 때만 넘겨, temperature를 받지 않는 구현도 그대로 감쌀 수 있게 한다.
        options = {} if temperature is None else {"temperature": temperature}
        with stage(name, images=len(image_urls)):
            return await self.inner.chat(
                messages, image_urls, max_tokens, response_format, **options
            )


class TracedEmbedder:
    def __init__(self, inner):
        self.inner = inner

    async def encode_text(self, text):
        with stage("embed"):
            return await self.inner.encode_text(text)


class TracedVectors:
    def __init__(self, inner):
        self.inner = inner

    async def search(self, vector, *, candidate_ids, limit):
        with stage("retrieve"):
            hits = await self.inner.search(vector, candidate_ids=candidate_ids, limit=limit)
        # 컷오프를 조정할 근거. 컷오프 후 개수는 ai_query의 returned_count에 남는다.
        if trace := current_trace():
            scores = [hit.score for hit in hits]
            trace.fields.update(
                hit_count=len(hits),
                top_score=round(max(scores), 4) if scores else None,
                min_score=round(min(scores), 4) if scores else None,
            )
        return hits


class TracedImages:
    def __init__(self, inner):
        self.inner = inner

    async def resolve(self, storage_key):
        with stage("sign_url"):
            return await self.inner.resolve(storage_key)
