from pyfedappwrap.engine.runtime import FedDBEngine

from app import ComparativeClusteringBenchmarkApp

engine = FedDBEngine()
engine.register(ComparativeClusteringBenchmarkApp())

if __name__ == '__main__':
    engine.start()
    engine.wait_until_stop()
