"""Source-reviewed imported serving protocols, distinct from trainable recipes."""
from dataclasses import dataclass


def is_imported(manifest):
    return manifest.get('origin') == 'imported'


@dataclass(frozen=True)
class ImportedSpec:
    def readiness_input(self, dimensions):
        from .pretrained_yolox import probe_input
        return probe_input()

    def validate_readiness(self, predictions, manifest=None):
        from .pretrained_yolox import probe_output
        return probe_output(predictions)

    def validate_features(self, features, dimensions=None):
        from .pretrained_yolox import validate_features
        return validate_features(features, dimensions)

    def load(self, path, expected_hash):
        from .pretrained_yolox import load
        return load(path, expected_hash)

    def predict(self, model, features):
        from .pretrained_yolox import predict
        return predict(model, features)


def for_manifest(manifest):
    if is_imported(manifest):
        from .pretrained_yolox import PROTOCOL, CONTRACT, FORMAT
        if (manifest.get('protocol') != PROTOCOL or manifest.get('contract') != CONTRACT
                or manifest.get('format') != FORMAT
                or any(manifest.get(key) is not None for key in
                       ('recipe', 'run_id', 'attempt_id', 'evaluation_id', 'evaluation'))):
            raise ValueError('Imported serving identity is inconsistent')
        return ImportedSpec()
    if manifest.get('origin') not in (None, 'trained'):
        raise ValueError('Unknown model origin')
    from .recipes import for_manifest as trained_spec
    return trained_spec(manifest)


def artifact_path(runtime, manifest):
    if is_imported(manifest):
        from .model_imports import artifact_path as imported_path
        return imported_path(runtime, manifest['artifact_id'])
    if manifest['artifact_id'] != manifest['attempt_id']:
        raise ValueError('Invalid artifact binding')
    return runtime.artifact(manifest['attempt_id'])


def public_identity(manifest):
    if is_imported(manifest):
        from .model_imports import public_identity as imported_identity
        return imported_identity(manifest)
    if manifest.get('recipe') == 'yolox-nano-dataset/v1':
        from .dataset_serving import public_identity as dataset_identity
        return dataset_identity(manifest)
    return {}


def validate_loaded_identity(manifest, model):
    keys = ('protocol', 'contract', 'format', 'dimensions') if is_imported(manifest) else (
        'recipe', 'contract', 'format', 'dimensions')
    if any(model.get(key) != manifest.get(key) for key in keys):
        raise ValueError('Model manifest contract mismatch')
    if is_imported(manifest):
        from .pretrained_yolox import ASSET_SHA256, ASSET_BYTES, CLASS_NAMES
        if (manifest.get('artifact_sha256') != ASSET_SHA256
                or manifest.get('artifact_bytes') != ASSET_BYTES
                or manifest.get('class_names') != list(CLASS_NAMES)
                or model.get('class_names') != list(CLASS_NAMES)
                or model.get('validation', {}).get('sha256') != ASSET_SHA256
                or manifest.get('local_evaluation_status') != 'not_evaluated'
                or manifest.get('production_allowed') is not False
                or manifest.get('device_certified') is not False):
            raise ValueError('Imported model binding mismatch')
    elif manifest.get('recipe') == 'yolox-nano-dataset/v1':
        from .dataset_serving import validate_identity
        validate_identity(manifest, model)
