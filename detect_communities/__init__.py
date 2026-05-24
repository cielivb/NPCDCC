from .pipeline import run

# The only exposed function will be detect_communities.run(), which is an
# alias for pipeline.run()
__all__ = ["run"]