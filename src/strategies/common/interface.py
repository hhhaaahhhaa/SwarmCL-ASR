class IParticle(object):
    @classmethod
    def dummy(cls) -> "IParticle":
        raise NotImplementedError

    def cache(self, tag: str) -> None:
        raise NotImplementedError
