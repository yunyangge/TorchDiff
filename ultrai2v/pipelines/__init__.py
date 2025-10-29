from .t2v_pipeline import pipelines as t2v_pipelines

pipelines = {}
pipelines.update(t2v_pipelines)

__all__ = [
    'pipelines'
]