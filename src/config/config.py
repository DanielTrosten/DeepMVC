import pickle
from pydantic import BaseModel, validator


class Config(BaseModel):
    illegal_vars = ["illegal_vars", "_glob_vars", "class_name"]
    _glob_vars = tuple()
    class_name: str = None

    class Config:
        validate_all = True
        extra = "forbid"

    @validator("class_name", pre=True, always=True)
    def set_class_name(cls, v):
        assert v is None, "Config property 'class_name' is not settable."
        return cls.__name__

    def set_globs(self, globs=None):
        if globs is None:
            globs = {}
        else:
            for key, value in globs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        for k in self._glob_vars:
            globs[k] = getattr(self, k)

        for _, value in self:
            if isinstance(value, Config):
                value.set_globs(globs=globs)

    def to_pickle(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self, f)
