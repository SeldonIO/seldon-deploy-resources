import kfp.dsl as dsl
import json
import kfp.components as comp
from collections import OrderedDict
from kubernetes import client as k8s_client


def setup(DEPLOY_PASSWORD: str, DEPLOY_SERVER: str, DEPLOY_USER: str, MINIO_ACCESS_KEY: str, MINIO_HOST: str, MINIO_MODEL_BUCKET: str, MINIO_SECRET_KEY: str):
    pipeline_parameters_block = '''
    DEPLOY_PASSWORD = "{}"
    DEPLOY_SERVER = "{}"
    DEPLOY_USER = "{}"
    MINIO_ACCESS_KEY = "{}"
    MINIO_HOST = "{}"
    MINIO_MODEL_BUCKET = "{}"
    MINIO_SECRET_KEY = "{}"
    '''.format(DEPLOY_PASSWORD, DEPLOY_SERVER, DEPLOY_USER, MINIO_ACCESS_KEY, MINIO_HOST, MINIO_MODEL_BUCKET, MINIO_SECRET_KEY)

    from kale.utils import mlmd_utils as _kale_mlmd_utils
    _kale_mlmd_utils.init_metadata()

    block1 = '''
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from alibi.explainers import AnchorTabular
    from alibi.datasets import fetch_adult
    from minio import Minio
    from minio.error import ResponseError
    from joblib import dump, load
    import dill
    import time
    import json
    from subprocess import run, Popen, PIPE
    from alibi_detect.utils.data import create_outlier_batch
    import swagger_client
    from swagger_client.rest import ApiException
    import yaml
    import json
    import urllib3
    urllib3.disable_warnings()
    '''

    block2 = '''
    def get_minio():
        return Minio(MINIO_HOST,
                        access_key=MINIO_ACCESS_KEY,
                        secret_key=MINIO_SECRET_KEY,
                        secure=False)
    '''

    block3 = '''
    def get_swagger_configuration():
        configuration = swagger_client.Configuration()
        configuration.host = 'http://seldon-deploy.seldon-system/seldon-deploy/api/v1alpha1'
        return configuration
    '''

    block4 = '''
    import requests

    from urllib.parse import urlparse

    KF_SESSION_COOKIE_NAME = "authservice_session"


    class SessionAuthenticator:
        """
        Returns the cookie token.
        """

        def __init__(self, server: str):
            self._server = server

            url = urlparse(server)
            self._host = f"{url.scheme}://{url.netloc}"

        def authenticate(self, user: str, password: str) -> str:
            auth_path = self._get_auth_path()
            success_path = self._submit_auth(auth_path, user, password)
            session_cookie = self._get_session_cookie(success_path)
            return session_cookie

        def _get_auth_path(self) -> str:
            # Send unauthenticated request
            res = requests.get(self._server, allow_redirects=False, verify=False)

            # Follow the 302 redirect
            oidc_path = res.headers["Location"]
            oidc_endpoint = f"{self._host}{oidc_path}"
            res = requests.get(oidc_endpoint, allow_redirects=False, verify=False)

            return res.headers["Location"]

        def _submit_auth(self, auth_path: str, user: str, password: str) -> str:
            auth_endpoint = f"{self._host}{auth_path}"
            auth_payload = {"login": user, "password": password}
            res = requests.post(auth_endpoint, auth_payload, allow_redirects=False, verify=False)
            
            login_path = res.headers["Location"]
            login_endpoint = f"{self._host}{login_path}"
            res = requests.get(login_endpoint, allow_redirects=False, verify=False)

            return res.headers["Location"]

        def _get_session_cookie(self, success_path: str) -> str:
            success_endpoint = f"{self._host}{success_path}"
            res = requests.get(success_endpoint, allow_redirects=False, verify=False)
            print(res.cookies)
            return res.cookies[KF_SESSION_COOKIE_NAME]

    def authenticate():
        authenticator = SessionAuthenticator(DEPLOY_SERVER)

        cookie = authenticator.authenticate(DEPLOY_USER, DEPLOY_PASSWORD)
        return cookie
    '''

    block5 = '''
    minioClient = get_minio()
    buckets = minioClient.list_buckets()
    for bucket in buckets:
        print(bucket.name, bucket.creation_date)
    '''

    block6 = '''
    if not minioClient.bucket_exists(MINIO_MODEL_BUCKET):
        minioClient.make_bucket(MINIO_MODEL_BUCKET)
    '''

    # run the code blocks inside a jupyter kernel
    from kale.utils.jupyter_utils import run_code as _kale_run_code
    from kale.utils.kfp_utils import \
        update_uimetadata as _kale_update_uimetadata
    blocks = (pipeline_parameters_block,
              block1,
              block2,
              block3,
              block4,
              block5,
              block6,
              )
    html_artifact = _kale_run_code(blocks)
    with open("/setup.html", "w") as f:
        f.write(html_artifact)
    _kale_update_uimetadata('setup')

    _kale_mlmd_utils.call("mark_execution_complete")


def build_model(DEPLOY_PASSWORD: str, DEPLOY_SERVER: str, DEPLOY_USER: str, INCOME_MODEL_PATH: str, MINIO_ACCESS_KEY: str, MINIO_HOST: str, MINIO_MODEL_BUCKET: str, MINIO_SECRET_KEY: str):
    pipeline_parameters_block = '''
    DEPLOY_PASSWORD = "{}"
    DEPLOY_SERVER = "{}"
    DEPLOY_USER = "{}"
    INCOME_MODEL_PATH = "{}"
    MINIO_ACCESS_KEY = "{}"
    MINIO_HOST = "{}"
    MINIO_MODEL_BUCKET = "{}"
    MINIO_SECRET_KEY = "{}"
    '''.format(DEPLOY_PASSWORD, DEPLOY_SERVER, DEPLOY_USER, INCOME_MODEL_PATH, MINIO_ACCESS_KEY, MINIO_HOST, MINIO_MODEL_BUCKET, MINIO_SECRET_KEY)

    from kale.utils import mlmd_utils as _kale_mlmd_utils
    _kale_mlmd_utils.init_metadata()

    block1 = '''
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from alibi.explainers import AnchorTabular
    from alibi.datasets import fetch_adult
    from minio import Minio
    from minio.error import ResponseError
    from joblib import dump, load
    import dill
    import time
    import json
    from subprocess import run, Popen, PIPE
    from alibi_detect.utils.data import create_outlier_batch
    import swagger_client
    from swagger_client.rest import ApiException
    import yaml
    import json
    import urllib3
    urllib3.disable_warnings()
    '''

    block2 = '''
    def get_minio():
        return Minio(MINIO_HOST,
                        access_key=MINIO_ACCESS_KEY,
                        secret_key=MINIO_SECRET_KEY,
                        secure=False)
    '''

    block3 = '''
    def get_swagger_configuration():
        configuration = swagger_client.Configuration()
        configuration.host = 'http://seldon-deploy.seldon-system/seldon-deploy/api/v1alpha1'
        return configuration
    '''

    block4 = '''
    import requests

    from urllib.parse import urlparse

    KF_SESSION_COOKIE_NAME = "authservice_session"


    class SessionAuthenticator:
        """
        Returns the cookie token.
        """

        def __init__(self, server: str):
            self._server = server

            url = urlparse(server)
            self._host = f"{url.scheme}://{url.netloc}"

        def authenticate(self, user: str, password: str) -> str:
            auth_path = self._get_auth_path()
            success_path = self._submit_auth(auth_path, user, password)
            session_cookie = self._get_session_cookie(success_path)
            return session_cookie

        def _get_auth_path(self) -> str:
            # Send unauthenticated request
            res = requests.get(self._server, allow_redirects=False, verify=False)

            # Follow the 302 redirect
            oidc_path = res.headers["Location"]
            oidc_endpoint = f"{self._host}{oidc_path}"
            res = requests.get(oidc_endpoint, allow_redirects=False, verify=False)

            return res.headers["Location"]

        def _submit_auth(self, auth_path: str, user: str, password: str) -> str:
            auth_endpoint = f"{self._host}{auth_path}"
            auth_payload = {"login": user, "password": password}
            res = requests.post(auth_endpoint, auth_payload, allow_redirects=False, verify=False)
            
            login_path = res.headers["Location"]
            login_endpoint = f"{self._host}{login_path}"
            res = requests.get(login_endpoint, allow_redirects=False, verify=False)

            return res.headers["Location"]

        def _get_session_cookie(self, success_path: str) -> str:
            success_endpoint = f"{self._host}{success_path}"
            res = requests.get(success_endpoint, allow_redirects=False, verify=False)
            print(res.cookies)
            return res.cookies[KF_SESSION_COOKIE_NAME]

    def authenticate():
        authenticator = SessionAuthenticator(DEPLOY_SERVER)

        cookie = authenticator.authenticate(DEPLOY_USER, DEPLOY_PASSWORD)
        return cookie
    '''

    block5 = '''
    adult = fetch_adult()
    adult.keys()
    '''

    block6 = '''
    data = adult.data
    target = adult.target
    feature_names = adult.feature_names
    category_map = adult.category_map
    '''

    block7 = '''
    from alibi.utils.data import gen_category_map
    '''

    block8 = '''
    np.random.seed(0)
    data_perm = np.random.permutation(np.c_[data, target])
    data = data_perm[:,:-1]
    target = data_perm[:,-1]
    '''

    block9 = '''
    idx = 30000
    X_train,Y_train = data[:idx,:], target[:idx]
    X_test, Y_test = data[idx+1:,:], target[idx+1:]
    '''

    block10 = '''
    ordinal_features = [x for x in range(len(feature_names)) if x not in list(category_map.keys())]
    ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                          ('scaler', StandardScaler())])
    '''

    block11 = '''
    categorical_features = list(category_map.keys())
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    '''

    block12 = '''
    preprocessor = ColumnTransformer(transformers=[('num', ordinal_transformer, ordinal_features),
                                                   ('cat', categorical_transformer, categorical_features)])
    '''

    block13 = '''
    np.random.seed(0)
    clf = RandomForestClassifier(n_estimators=50)
    '''

    block14 = '''
    model=Pipeline(steps=[("preprocess",preprocessor),("model",clf)])
    model.fit(X_train,Y_train)
    '''

    block15 = '''
    def predict_fn(x):
        return model.predict(x)
    '''

    block16 = '''
    #predict_fn = lambda x: clf.predict(preprocessor.transform(x))
    print('Train accuracy: ', accuracy_score(Y_train, predict_fn(X_train)))
    print('Test accuracy: ', accuracy_score(Y_test, predict_fn(X_test)))
    '''

    block17 = '''
    dump(model, 'model.joblib') 
    '''

    block18 = '''
    print(get_minio().fput_object(MINIO_MODEL_BUCKET, f"{INCOME_MODEL_PATH}/model.joblib", 'model.joblib'))
    '''

    data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale.marshal import utils as _kale_marshal_utils
    _kale_marshal_utils.set_kale_data_directory("/marshal")
    _kale_marshal_utils.save(X_test, "X_test")
    _kale_marshal_utils.save(X_train, "X_train")
    _kale_marshal_utils.save(Y_train, "Y_train")
    _kale_marshal_utils.save(adult, "adult")
    _kale_marshal_utils.save(category_map, "category_map")
    _kale_marshal_utils.save(feature_names, "feature_names")
    _kale_marshal_utils.save(model, "model")
    _kale_marshal_utils.save(predict_fn, "predict_fn")
    # -----------------------DATA SAVING END-----------------------------------
    '''

    # run the code blocks inside a jupyter kernel
    from kale.utils.jupyter_utils import run_code as _kale_run_code
    from kale.utils.kfp_utils import \
        update_uimetadata as _kale_update_uimetadata
    blocks = (pipeline_parameters_block,
              block1,
              block2,
              block3,
              block4,
              block5,
              block6,
              block7,
              block8,
              block9,
              block10,
              block11,
              block12,
              block13,
              block14,
              block15,
              block16,
              block17,
              block18,
              data_saving_block)
    html_artifact = _kale_run_code(blocks)
    with open("/build_model.html", "w") as f:
        f.write(html_artifact)
    _kale_update_uimetadata('build_model')

    _kale_mlmd_utils.call("mark_execution_complete")


def build_outlier(DEPLOY_PASSWORD: str, DEPLOY_SERVER: str, DEPLOY_USER: str, MINIO_ACCESS_KEY: str, MINIO_HOST: str, MINIO_MODEL_BUCKET: str, MINIO_SECRET_KEY: str, OUTLIER_MODEL_PATH: str):
    pipeline_parameters_block = '''
    DEPLOY_PASSWORD = "{}"
    DEPLOY_SERVER = "{}"
    DEPLOY_USER = "{}"
    MINIO_ACCESS_KEY = "{}"
    MINIO_HOST = "{}"
    MINIO_MODEL_BUCKET = "{}"
    MINIO_SECRET_KEY = "{}"
    OUTLIER_MODEL_PATH = "{}"
    '''.format(DEPLOY_PASSWORD, DEPLOY_SERVER, DEPLOY_USER, MINIO_ACCESS_KEY, MINIO_HOST, MINIO_MODEL_BUCKET, MINIO_SECRET_KEY, OUTLIER_MODEL_PATH)

    from kale.utils import mlmd_utils as _kale_mlmd_utils
    _kale_mlmd_utils.init_metadata()

    data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale.marshal import utils as _kale_marshal_utils
    _kale_marshal_utils.set_kale_data_directory("/marshal")
    _kale_marshal_utils.set_kale_directory_file_names()
    X_train = _kale_marshal_utils.load("X_train")
    Y_train = _kale_marshal_utils.load("Y_train")
    # -----------------------DATA LOADING END----------------------------------
    '''

    block1 = '''
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from alibi.explainers import AnchorTabular
    from alibi.datasets import fetch_adult
    from minio import Minio
    from minio.error import ResponseError
    from joblib import dump, load
    import dill
    import time
    import json
    from subprocess import run, Popen, PIPE
    from alibi_detect.utils.data import create_outlier_batch
    import swagger_client
    from swagger_client.rest import ApiException
    import yaml
    import json
    import urllib3
    urllib3.disable_warnings()
    '''

    block2 = '''
    def get_minio():
        return Minio(MINIO_HOST,
                        access_key=MINIO_ACCESS_KEY,
                        secret_key=MINIO_SECRET_KEY,
                        secure=False)
    '''

    block3 = '''
    def get_swagger_configuration():
        configuration = swagger_client.Configuration()
        configuration.host = 'http://seldon-deploy.seldon-system/seldon-deploy/api/v1alpha1'
        return configuration
    '''

    block4 = '''
    import requests

    from urllib.parse import urlparse

    KF_SESSION_COOKIE_NAME = "authservice_session"


    class SessionAuthenticator:
        """
        Returns the cookie token.
        """

        def __init__(self, server: str):
            self._server = server

            url = urlparse(server)
            self._host = f"{url.scheme}://{url.netloc}"

        def authenticate(self, user: str, password: str) -> str:
            auth_path = self._get_auth_path()
            success_path = self._submit_auth(auth_path, user, password)
            session_cookie = self._get_session_cookie(success_path)
            return session_cookie

        def _get_auth_path(self) -> str:
            # Send unauthenticated request
            res = requests.get(self._server, allow_redirects=False, verify=False)

            # Follow the 302 redirect
            oidc_path = res.headers["Location"]
            oidc_endpoint = f"{self._host}{oidc_path}"
            res = requests.get(oidc_endpoint, allow_redirects=False, verify=False)

            return res.headers["Location"]

        def _submit_auth(self, auth_path: str, user: str, password: str) -> str:
            auth_endpoint = f"{self._host}{auth_path}"
            auth_payload = {"login": user, "password": password}
            res = requests.post(auth_endpoint, auth_payload, allow_redirects=False, verify=False)
            
            login_path = res.headers["Location"]
            login_endpoint = f"{self._host}{login_path}"
            res = requests.get(login_endpoint, allow_redirects=False, verify=False)

            return res.headers["Location"]

        def _get_session_cookie(self, success_path: str) -> str:
            success_endpoint = f"{self._host}{success_path}"
            res = requests.get(success_endpoint, allow_redirects=False, verify=False)
            print(res.cookies)
            return res.cookies[KF_SESSION_COOKIE_NAME]

    def authenticate():
        authenticator = SessionAuthenticator(DEPLOY_SERVER)

        cookie = authenticator.authenticate(DEPLOY_USER, DEPLOY_PASSWORD)
        return cookie
    '''

    block5 = '''
    from alibi_detect.od import IForest

    od = IForest(
        threshold=0.,
        n_estimators=200,
    )
    '''

    block6 = '''
    od.fit(X_train)
    '''

    block7 = '''
    np.random.seed(0)
    perc_outlier = 5
    threshold_batch = create_outlier_batch(X_train, Y_train, n_samples=1000, perc_outlier=perc_outlier)
    X_threshold, y_threshold = threshold_batch.data.astype('float'), threshold_batch.target
    #X_threshold = (X_threshold - mean) / stdev
    print('{}% outliers'.format(100 * y_threshold.mean()))
    '''

    block8 = '''
    od.infer_threshold(X_threshold, threshold_perc=100-perc_outlier)
    print('New threshold: {}'.format(od.threshold))
    threshold = od.threshold
    '''

    block9 = '''
    X_outlier = [[300,  4,  4,  2,  1,  4,  4,  0,  0,  0, 600,  9]]
    '''

    block10 = '''
    od.predict(
        X_outlier
    )
    '''

    block11 = '''
    from alibi_detect.utils.saving import save_detector, load_detector
    from os import listdir
    from os.path import isfile, join

    filepath="ifoutlier"
    save_detector(od, filepath) 
    onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    for filename in onlyfiles:
        print(filename)
        print(get_minio().fput_object(MINIO_MODEL_BUCKET, f"{OUTLIER_MODEL_PATH}/{filename}", join(filepath, filename)))
    '''

    # run the code blocks inside a jupyter kernel
    from kale.utils.jupyter_utils import run_code as _kale_run_code
    from kale.utils.kfp_utils import \
        update_uimetadata as _kale_update_uimetadata
    blocks = (pipeline_parameters_block, data_loading_block,
              block1,
              block2,
              block3,
              block4,
              block5,
              block6,
              block7,
              block8,
              block9,
              block10,
              block11,
              )
    html_artifact = _kale_run_code(blocks)
    with open("/build_outlier.html", "w") as f:
        f.write(html_artifact)
    _kale_update_uimetadata('build_outlier')

    _kale_mlmd_utils.call("mark_execution_complete")


def train_explainer(DEPLOY_PASSWORD: str, DEPLOY_SERVER: str, DEPLOY_USER: str, EXPLAINER_MODEL_PATH: str, MINIO_ACCESS_KEY: str, MINIO_HOST: str, MINIO_MODEL_BUCKET: str, MINIO_SECRET_KEY: str):
    pipeline_parameters_block = '''
    DEPLOY_PASSWORD = "{}"
    DEPLOY_SERVER = "{}"
    DEPLOY_USER = "{}"
    EXPLAINER_MODEL_PATH = "{}"
    MINIO_ACCESS_KEY = "{}"
    MINIO_HOST = "{}"
    MINIO_MODEL_BUCKET = "{}"
    MINIO_SECRET_KEY = "{}"
    '''.format(DEPLOY_PASSWORD, DEPLOY_SERVER, DEPLOY_USER, EXPLAINER_MODEL_PATH, MINIO_ACCESS_KEY, MINIO_HOST, MINIO_MODEL_BUCKET, MINIO_SECRET_KEY)

    from kale.utils import mlmd_utils as _kale_mlmd_utils
    _kale_mlmd_utils.init_metadata()

    data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale.marshal import utils as _kale_marshal_utils
    _kale_marshal_utils.set_kale_data_directory("/marshal")
    _kale_marshal_utils.set_kale_directory_file_names()
    X_train = _kale_marshal_utils.load("X_train")
    category_map = _kale_marshal_utils.load("category_map")
    feature_names = _kale_marshal_utils.load("feature_names")
    model = _kale_marshal_utils.load("model")
    predict_fn = _kale_marshal_utils.load("predict_fn")
    # -----------------------DATA LOADING END----------------------------------
    '''

    block1 = '''
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from alibi.explainers import AnchorTabular
    from alibi.datasets import fetch_adult
    from minio import Minio
    from minio.error import ResponseError
    from joblib import dump, load
    import dill
    import time
    import json
    from subprocess import run, Popen, PIPE
    from alibi_detect.utils.data import create_outlier_batch
    import swagger_client
    from swagger_client.rest import ApiException
    import yaml
    import json
    import urllib3
    urllib3.disable_warnings()
    '''

    block2 = '''
    def get_minio():
        return Minio(MINIO_HOST,
                        access_key=MINIO_ACCESS_KEY,
                        secret_key=MINIO_SECRET_KEY,
                        secure=False)
    '''

    block3 = '''
    def get_swagger_configuration():
        configuration = swagger_client.Configuration()
        configuration.host = 'http://seldon-deploy.seldon-system/seldon-deploy/api/v1alpha1'
        return configuration
    '''

    block4 = '''
    import requests

    from urllib.parse import urlparse

    KF_SESSION_COOKIE_NAME = "authservice_session"


    class SessionAuthenticator:
        """
        Returns the cookie token.
        """

        def __init__(self, server: str):
            self._server = server

            url = urlparse(server)
            self._host = f"{url.scheme}://{url.netloc}"

        def authenticate(self, user: str, password: str) -> str:
            auth_path = self._get_auth_path()
            success_path = self._submit_auth(auth_path, user, password)
            session_cookie = self._get_session_cookie(success_path)
            return session_cookie

        def _get_auth_path(self) -> str:
            # Send unauthenticated request
            res = requests.get(self._server, allow_redirects=False, verify=False)

            # Follow the 302 redirect
            oidc_path = res.headers["Location"]
            oidc_endpoint = f"{self._host}{oidc_path}"
            res = requests.get(oidc_endpoint, allow_redirects=False, verify=False)

            return res.headers["Location"]

        def _submit_auth(self, auth_path: str, user: str, password: str) -> str:
            auth_endpoint = f"{self._host}{auth_path}"
            auth_payload = {"login": user, "password": password}
            res = requests.post(auth_endpoint, auth_payload, allow_redirects=False, verify=False)
            
            login_path = res.headers["Location"]
            login_endpoint = f"{self._host}{login_path}"
            res = requests.get(login_endpoint, allow_redirects=False, verify=False)

            return res.headers["Location"]

        def _get_session_cookie(self, success_path: str) -> str:
            success_endpoint = f"{self._host}{success_path}"
            res = requests.get(success_endpoint, allow_redirects=False, verify=False)
            print(res.cookies)
            return res.cookies[KF_SESSION_COOKIE_NAME]

    def authenticate():
        authenticator = SessionAuthenticator(DEPLOY_SERVER)

        cookie = authenticator.authenticate(DEPLOY_USER, DEPLOY_PASSWORD)
        return cookie
    '''

    block5 = '''
    model.predict(X_train)
    explainer = AnchorTabular(predict_fn, feature_names, categorical_names=category_map)
    '''

    block6 = '''
    explainer.fit(X_train, disc_perc=[25, 50, 75])
    '''

    block7 = '''
    with open("explainer.dill", "wb") as dill_file:
        dill.dump(explainer, dill_file)    
        dill_file.close()
    print(get_minio().fput_object(MINIO_MODEL_BUCKET, f"{EXPLAINER_MODEL_PATH}/explainer.dill", 'explainer.dill'))
    '''

    data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale.marshal import utils as _kale_marshal_utils
    _kale_marshal_utils.set_kale_data_directory("/marshal")
    _kale_marshal_utils.save(X_train, "X_train")
    _kale_marshal_utils.save(explainer, "explainer")
    _kale_marshal_utils.save(model, "model")
    # -----------------------DATA SAVING END-----------------------------------
    '''

    # run the code blocks inside a jupyter kernel
    from kale.utils.jupyter_utils import run_code as _kale_run_code
    from kale.utils.kfp_utils import \
        update_uimetadata as _kale_update_uimetadata
    blocks = (pipeline_parameters_block, data_loading_block,
              block1,
              block2,
              block3,
              block4,
              block5,
              block6,
              block7,
              data_saving_block)
    html_artifact = _kale_run_code(blocks)
    with open("/train_explainer.html", "w") as f:
        f.write(html_artifact)
    _kale_update_uimetadata('train_explainer')

    _kale_mlmd_utils.call("mark_execution_complete")


def deploy_seldon(DEPLOY_NAMESPACE: str, DEPLOY_PASSWORD: str, DEPLOY_SERVER: str, DEPLOY_USER: str, EXPLAINER_MODEL_PATH: str, INCOME_MODEL_PATH: str, MINIO_ACCESS_KEY: str, MINIO_HOST: str, MINIO_MODEL_BUCKET: str, MINIO_SECRET_KEY: str):
    pipeline_parameters_block = '''
    DEPLOY_NAMESPACE = "{}"
    DEPLOY_PASSWORD = "{}"
    DEPLOY_SERVER = "{}"
    DEPLOY_USER = "{}"
    EXPLAINER_MODEL_PATH = "{}"
    INCOME_MODEL_PATH = "{}"
    MINIO_ACCESS_KEY = "{}"
    MINIO_HOST = "{}"
    MINIO_MODEL_BUCKET = "{}"
    MINIO_SECRET_KEY = "{}"
    '''.format(DEPLOY_NAMESPACE, DEPLOY_PASSWORD, DEPLOY_SERVER, DEPLOY_USER, EXPLAINER_MODEL_PATH, INCOME_MODEL_PATH, MINIO_ACCESS_KEY, MINIO_HOST, MINIO_MODEL_BUCKET, MINIO_SECRET_KEY)

    from kale.utils import mlmd_utils as _kale_mlmd_utils
    _kale_mlmd_utils.init_metadata()

    block1 = '''
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from alibi.explainers import AnchorTabular
    from alibi.datasets import fetch_adult
    from minio import Minio
    from minio.error import ResponseError
    from joblib import dump, load
    import dill
    import time
    import json
    from subprocess import run, Popen, PIPE
    from alibi_detect.utils.data import create_outlier_batch
    import swagger_client
    from swagger_client.rest import ApiException
    import yaml
    import json
    import urllib3
    urllib3.disable_warnings()
    '''

    block2 = '''
    def get_minio():
        return Minio(MINIO_HOST,
                        access_key=MINIO_ACCESS_KEY,
                        secret_key=MINIO_SECRET_KEY,
                        secure=False)
    '''

    block3 = '''
    def get_swagger_configuration():
        configuration = swagger_client.Configuration()
        configuration.host = 'http://seldon-deploy.seldon-system/seldon-deploy/api/v1alpha1'
        return configuration
    '''

    block4 = '''
    import requests

    from urllib.parse import urlparse

    KF_SESSION_COOKIE_NAME = "authservice_session"


    class SessionAuthenticator:
        """
        Returns the cookie token.
        """

        def __init__(self, server: str):
            self._server = server

            url = urlparse(server)
            self._host = f"{url.scheme}://{url.netloc}"

        def authenticate(self, user: str, password: str) -> str:
            auth_path = self._get_auth_path()
            success_path = self._submit_auth(auth_path, user, password)
            session_cookie = self._get_session_cookie(success_path)
            return session_cookie

        def _get_auth_path(self) -> str:
            # Send unauthenticated request
            res = requests.get(self._server, allow_redirects=False, verify=False)

            # Follow the 302 redirect
            oidc_path = res.headers["Location"]
            oidc_endpoint = f"{self._host}{oidc_path}"
            res = requests.get(oidc_endpoint, allow_redirects=False, verify=False)

            return res.headers["Location"]

        def _submit_auth(self, auth_path: str, user: str, password: str) -> str:
            auth_endpoint = f"{self._host}{auth_path}"
            auth_payload = {"login": user, "password": password}
            res = requests.post(auth_endpoint, auth_payload, allow_redirects=False, verify=False)
            
            login_path = res.headers["Location"]
            login_endpoint = f"{self._host}{login_path}"
            res = requests.get(login_endpoint, allow_redirects=False, verify=False)

            return res.headers["Location"]

        def _get_session_cookie(self, success_path: str) -> str:
            success_endpoint = f"{self._host}{success_path}"
            res = requests.get(success_endpoint, allow_redirects=False, verify=False)
            print(res.cookies)
            return res.cookies[KF_SESSION_COOKIE_NAME]

    def authenticate():
        authenticator = SessionAuthenticator(DEPLOY_SERVER)

        cookie = authenticator.authenticate(DEPLOY_USER, DEPLOY_PASSWORD)
        return cookie
    '''

    block5 = '''
    secret = f"""apiVersion: v1
    kind: Secret
    metadata:
      name: seldon-init-container-secret
      namespace: {DEPLOY_NAMESPACE}
    type: Opaque
    stringData:
      AWS_ACCESS_KEY_ID: {MINIO_ACCESS_KEY}
      AWS_SECRET_ACCESS_KEY: {MINIO_SECRET_KEY}
      AWS_ENDPOINT_URL: http://{MINIO_HOST}
      USE_SSL: "false"
    """
    with open("secret.yaml","w") as f:
        f.write(secret)
    run("cat secret.yaml | kubectl apply -f -", shell=True)
    '''

    block6 = '''
    sa = f"""apiVersion: v1
    kind: ServiceAccount
    metadata:
      name: minio-sa
      namespace: {DEPLOY_NAMESPACE}
    secrets:
      - name: seldon-init-container-secret
    """
    with open("sa.yaml","w") as f:
        f.write(sa)
    run("kubectl apply -f sa.yaml", shell=True)
    '''

    block7 = '''
    configuration = get_swagger_configuration()
    # create an instance of the API class
    dep_instance = swagger_client.MLDeploymentsApi(swagger_client.ApiClient(configuration))
    namespace = 'admin' # str | Namespace provides a logical grouping of resources
    '''

    block8 = '''
    model_name = "income-classifier"
    model_yaml=f"""apiVersion: machinelearning.seldon.io/v1
    kind: SeldonDeployment
    metadata:
      name: {model_name}
      namespace: {DEPLOY_NAMESPACE}
    spec:
      predictors:
      - componentSpecs:
        graph:
          implementation: SKLEARN_SERVER
          modelUri: s3://{MINIO_MODEL_BUCKET}/{INCOME_MODEL_PATH}
          envSecretRefName: seldon-init-container-secret
          name: classifier
          logger:
             mode: all
             url: http://default-broker
        explainer:
          type: AnchorTabular
          modelUri: s3://{MINIO_MODEL_BUCKET}/{EXPLAINER_MODEL_PATH}
          envSecretRefName: seldon-init-container-secret
        name: default
        replicas: 1
    """
    d = yaml.safe_load(model_yaml)
    model_json = json.dumps(d)
    print(model_json)
    created = dep_instance.create_seldon_deployment(model_json, namespace)
    '''

    block9 = '''
    state = ""
    while not state == "Available":
        res = dep_instance.list_seldon_deployments(namespace)
        for sd in res.items:
            state = sd.status.state
            print(sd.status.state)
        time.sleep(2)
    time.sleep(10)
    '''

    block10 = '''
    cookie = authenticate()
    payload='{"data": {"ndarray": [[53,4,0,2,8,4,4,0,0,0,60,9]]}}'
    cookie_str = f"{KF_SESSION_COOKIE_NAME}={cookie}"
    predict_instance = swagger_client.PredictApi(swagger_client.ApiClient(configuration,cookie=cookie_str))
    prediction = predict_instance.predict_seldon_deployment(model_name,namespace, prediction={"data": {"ndarray": [[53,4,0,2,8,4,4,0,0,0,60,9]]}})
    print(prediction)
    '''

    block11 = '''
    explain_instance = swagger_client.ExplainerApi(swagger_client.ApiClient(configuration,cookie=cookie_str))
    tries = 0
    try:
        explanation = explain_instance.explain_seldon_deployment(namespace,model_name,explaindata={"data": {"ndarray": [[53,4,0,2,8,4,4,0,0,0,60,9]]}})
        print(explanation)
    except ApiException as e:
        print(e)
        if tries > 5:
            raise e
        print("Retrying")
        tries = tries +1
        time.sleep(5)
    '''

    data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale.marshal import utils as _kale_marshal_utils
    _kale_marshal_utils.set_kale_data_directory("/marshal")
    _kale_marshal_utils.save(model_name, "model_name")
    _kale_marshal_utils.save(namespace, "namespace")
    # -----------------------DATA SAVING END-----------------------------------
    '''

    # run the code blocks inside a jupyter kernel
    from kale.utils.jupyter_utils import run_code as _kale_run_code
    from kale.utils.kfp_utils import \
        update_uimetadata as _kale_update_uimetadata
    blocks = (pipeline_parameters_block,
              block1,
              block2,
              block3,
              block4,
              block5,
              block6,
              block7,
              block8,
              block9,
              block10,
              block11,
              data_saving_block)
    html_artifact = _kale_run_code(blocks)
    with open("/deploy_seldon.html", "w") as f:
        f.write(html_artifact)
    _kale_update_uimetadata('deploy_seldon')

    _kale_mlmd_utils.call("mark_execution_complete")


def deploy_outlier(DEPLOY_PASSWORD: str, DEPLOY_SERVER: str, DEPLOY_USER: str, MINIO_ACCESS_KEY: str, MINIO_HOST: str, MINIO_SECRET_KEY: str):
    pipeline_parameters_block = '''
    DEPLOY_PASSWORD = "{}"
    DEPLOY_SERVER = "{}"
    DEPLOY_USER = "{}"
    MINIO_ACCESS_KEY = "{}"
    MINIO_HOST = "{}"
    MINIO_SECRET_KEY = "{}"
    '''.format(DEPLOY_PASSWORD, DEPLOY_SERVER, DEPLOY_USER, MINIO_ACCESS_KEY, MINIO_HOST, MINIO_SECRET_KEY)

    from kale.utils import mlmd_utils as _kale_mlmd_utils
    _kale_mlmd_utils.init_metadata()

    data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale.marshal import utils as _kale_marshal_utils
    _kale_marshal_utils.set_kale_data_directory("/marshal")
    _kale_marshal_utils.set_kale_directory_file_names()
    model_name = _kale_marshal_utils.load("model_name")
    namespace = _kale_marshal_utils.load("namespace")
    # -----------------------DATA LOADING END----------------------------------
    '''

    block1 = '''
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from alibi.explainers import AnchorTabular
    from alibi.datasets import fetch_adult
    from minio import Minio
    from minio.error import ResponseError
    from joblib import dump, load
    import dill
    import time
    import json
    from subprocess import run, Popen, PIPE
    from alibi_detect.utils.data import create_outlier_batch
    import swagger_client
    from swagger_client.rest import ApiException
    import yaml
    import json
    import urllib3
    urllib3.disable_warnings()
    '''

    block2 = '''
    def get_minio():
        return Minio(MINIO_HOST,
                        access_key=MINIO_ACCESS_KEY,
                        secret_key=MINIO_SECRET_KEY,
                        secure=False)
    '''

    block3 = '''
    def get_swagger_configuration():
        configuration = swagger_client.Configuration()
        configuration.host = 'http://seldon-deploy.seldon-system/seldon-deploy/api/v1alpha1'
        return configuration
    '''

    block4 = '''
    import requests

    from urllib.parse import urlparse

    KF_SESSION_COOKIE_NAME = "authservice_session"


    class SessionAuthenticator:
        """
        Returns the cookie token.
        """

        def __init__(self, server: str):
            self._server = server

            url = urlparse(server)
            self._host = f"{url.scheme}://{url.netloc}"

        def authenticate(self, user: str, password: str) -> str:
            auth_path = self._get_auth_path()
            success_path = self._submit_auth(auth_path, user, password)
            session_cookie = self._get_session_cookie(success_path)
            return session_cookie

        def _get_auth_path(self) -> str:
            # Send unauthenticated request
            res = requests.get(self._server, allow_redirects=False, verify=False)

            # Follow the 302 redirect
            oidc_path = res.headers["Location"]
            oidc_endpoint = f"{self._host}{oidc_path}"
            res = requests.get(oidc_endpoint, allow_redirects=False, verify=False)

            return res.headers["Location"]

        def _submit_auth(self, auth_path: str, user: str, password: str) -> str:
            auth_endpoint = f"{self._host}{auth_path}"
            auth_payload = {"login": user, "password": password}
            res = requests.post(auth_endpoint, auth_payload, allow_redirects=False, verify=False)
            
            login_path = res.headers["Location"]
            login_endpoint = f"{self._host}{login_path}"
            res = requests.get(login_endpoint, allow_redirects=False, verify=False)

            return res.headers["Location"]

        def _get_session_cookie(self, success_path: str) -> str:
            success_endpoint = f"{self._host}{success_path}"
            res = requests.get(success_endpoint, allow_redirects=False, verify=False)
            print(res.cookies)
            return res.cookies[KF_SESSION_COOKIE_NAME]

    def authenticate():
        authenticator = SessionAuthenticator(DEPLOY_SERVER)

        cookie = authenticator.authenticate(DEPLOY_USER, DEPLOY_PASSWORD)
        return cookie
    '''

    block5 = '''
    configuration = get_swagger_configuration()
    outlier = swagger_client.OutlierDetectorApi(swagger_client.ApiClient(configuration))
    outlier_params = {
    "params": {
        "event_source": "io.seldon.serving.incomeod",
        "event_type": "io.seldon.serving.inference.outlier",
        "http_port": "8080",
        "model_name": "adultod",
        "protocol": "seldon.http",
        "reply_url": "http://default-broker",
        "storage_uri": "s3://seldon/sklearn/income/outlier",
        "env_secret_ref": "seldon-init-container-secret"
      }
    }
    res = outlier.create_outlier_detector_seldon_deployment(model_name, namespace, outlier_detector=outlier_params)
    '''

    data_saving_block = '''
    # -----------------------DATA SAVING START---------------------------------
    from kale.marshal import utils as _kale_marshal_utils
    _kale_marshal_utils.set_kale_data_directory("/marshal")
    _kale_marshal_utils.save(model_name, "model_name")
    _kale_marshal_utils.save(namespace, "namespace")
    # -----------------------DATA SAVING END-----------------------------------
    '''

    # run the code blocks inside a jupyter kernel
    from kale.utils.jupyter_utils import run_code as _kale_run_code
    from kale.utils.kfp_utils import \
        update_uimetadata as _kale_update_uimetadata
    blocks = (pipeline_parameters_block, data_loading_block,
              block1,
              block2,
              block3,
              block4,
              block5,
              data_saving_block)
    html_artifact = _kale_run_code(blocks)
    with open("/deploy_outlier.html", "w") as f:
        f.write(html_artifact)
    _kale_update_uimetadata('deploy_outlier')

    _kale_mlmd_utils.call("mark_execution_complete")


def deploy_event_display(DEPLOY_NAMESPACE: str, DEPLOY_PASSWORD: str, DEPLOY_SERVER: str, DEPLOY_USER: str, MINIO_ACCESS_KEY: str, MINIO_HOST: str, MINIO_SECRET_KEY: str):
    pipeline_parameters_block = '''
    DEPLOY_NAMESPACE = "{}"
    DEPLOY_PASSWORD = "{}"
    DEPLOY_SERVER = "{}"
    DEPLOY_USER = "{}"
    MINIO_ACCESS_KEY = "{}"
    MINIO_HOST = "{}"
    MINIO_SECRET_KEY = "{}"
    '''.format(DEPLOY_NAMESPACE, DEPLOY_PASSWORD, DEPLOY_SERVER, DEPLOY_USER, MINIO_ACCESS_KEY, MINIO_HOST, MINIO_SECRET_KEY)

    from kale.utils import mlmd_utils as _kale_mlmd_utils
    _kale_mlmd_utils.init_metadata()

    data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale.marshal import utils as _kale_marshal_utils
    _kale_marshal_utils.set_kale_data_directory("/marshal")
    _kale_marshal_utils.set_kale_directory_file_names()
    model_name = _kale_marshal_utils.load("model_name")
    namespace = _kale_marshal_utils.load("namespace")
    # -----------------------DATA LOADING END----------------------------------
    '''

    block1 = '''
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from alibi.explainers import AnchorTabular
    from alibi.datasets import fetch_adult
    from minio import Minio
    from minio.error import ResponseError
    from joblib import dump, load
    import dill
    import time
    import json
    from subprocess import run, Popen, PIPE
    from alibi_detect.utils.data import create_outlier_batch
    import swagger_client
    from swagger_client.rest import ApiException
    import yaml
    import json
    import urllib3
    urllib3.disable_warnings()
    '''

    block2 = '''
    def get_minio():
        return Minio(MINIO_HOST,
                        access_key=MINIO_ACCESS_KEY,
                        secret_key=MINIO_SECRET_KEY,
                        secure=False)
    '''

    block3 = '''
    def get_swagger_configuration():
        configuration = swagger_client.Configuration()
        configuration.host = 'http://seldon-deploy.seldon-system/seldon-deploy/api/v1alpha1'
        return configuration
    '''

    block4 = '''
    import requests

    from urllib.parse import urlparse

    KF_SESSION_COOKIE_NAME = "authservice_session"


    class SessionAuthenticator:
        """
        Returns the cookie token.
        """

        def __init__(self, server: str):
            self._server = server

            url = urlparse(server)
            self._host = f"{url.scheme}://{url.netloc}"

        def authenticate(self, user: str, password: str) -> str:
            auth_path = self._get_auth_path()
            success_path = self._submit_auth(auth_path, user, password)
            session_cookie = self._get_session_cookie(success_path)
            return session_cookie

        def _get_auth_path(self) -> str:
            # Send unauthenticated request
            res = requests.get(self._server, allow_redirects=False, verify=False)

            # Follow the 302 redirect
            oidc_path = res.headers["Location"]
            oidc_endpoint = f"{self._host}{oidc_path}"
            res = requests.get(oidc_endpoint, allow_redirects=False, verify=False)

            return res.headers["Location"]

        def _submit_auth(self, auth_path: str, user: str, password: str) -> str:
            auth_endpoint = f"{self._host}{auth_path}"
            auth_payload = {"login": user, "password": password}
            res = requests.post(auth_endpoint, auth_payload, allow_redirects=False, verify=False)
            
            login_path = res.headers["Location"]
            login_endpoint = f"{self._host}{login_path}"
            res = requests.get(login_endpoint, allow_redirects=False, verify=False)

            return res.headers["Location"]

        def _get_session_cookie(self, success_path: str) -> str:
            success_endpoint = f"{self._host}{success_path}"
            res = requests.get(success_endpoint, allow_redirects=False, verify=False)
            print(res.cookies)
            return res.cookies[KF_SESSION_COOKIE_NAME]

    def authenticate():
        authenticator = SessionAuthenticator(DEPLOY_SERVER)

        cookie = authenticator.authenticate(DEPLOY_USER, DEPLOY_PASSWORD)
        return cookie
    '''

    block5 = '''
    event_display=f"""apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: event-display
      namespace: {DEPLOY_NAMESPACE}          
    spec:
      replicas: 1
      selector:
        matchLabels: &labels
          app: event-display
      template:
        metadata:
          labels: *labels
        spec:
          containers:
            - name: helloworld-go
              # Source code: https://github.com/knative/eventing-contrib/tree/master/cmd/event_display
              image: gcr.io/knative-releases/knative.dev/eventing-contrib/cmd/event_display@sha256:f4628e97a836c77ed38bd3b6fd3d0b06de4d5e7db6704772fe674d48b20bd477
    ---
    kind: Service
    apiVersion: v1
    metadata:
      name: event-display
      namespace: {DEPLOY_NAMESPACE}
    spec:
      selector:
        app: event-display
      ports:
        - protocol: TCP
          port: 80
          targetPort: 8080
    ---
    apiVersion: eventing.knative.dev/v1alpha1
    kind: Trigger
    metadata:
      name: income-outlier-display
      namespace: {DEPLOY_NAMESPACE}
    spec:
      broker: default
      filter:
        attributes:
          type: io.seldon.serving.inference.outlier
      subscriber:
        ref:
          apiVersion: v1
          kind: Service
          name: event-display
    """
    with open("event_display.yaml","w") as f:
        f.write(event_display)
    run("kubectl apply -f event_display.yaml", shell=True)
    '''

    block6 = '''
    run(f"kubectl rollout status -n {DEPLOY_NAMESPACE} deploy/event-display -n {DEPLOY_NAMESPACE}", shell=True)
    '''

    block7 = '''
    def predict():
        configuration = get_swagger_configuration()
        cookie = authenticate()
        cookie_str = f"{KF_SESSION_COOKIE_NAME}={cookie}"
        predict_instance = swagger_client.PredictApi(swagger_client.ApiClient(configuration,cookie=cookie_str))
        prediction = predict_instance.predict_seldon_deployment(model_name,namespace, prediction={"data": {"ndarray": [[3000,4,4,2,1,4,4,0,0,0,600,9]]}})
        print(prediction)
    '''

    block8 = '''
    def get_outlier_event_display_logs():
        cmd=f"kubectl logs $(kubectl get pod -l app=event-display -o jsonpath='{{.items[0].metadata.name}}' -n {DEPLOY_NAMESPACE}) -n {DEPLOY_NAMESPACE}"
        ret = Popen(cmd, shell=True,stdout=PIPE)
        res = ret.stdout.read().decode("utf-8").split("\\n")
        data= []
        for i in range(0,len(res)):
            if res[i] == 'Data,':
                j = json.loads(json.loads(res[i+1]))
                print(j)
                if "is_outlier"in j["data"].keys():
                    data.append(j)
        if len(data) > 0:
            return data[-1]
        else:
            return None
    j = None
    while j is None:
        predict()
        print("Waiting for outlier logs, sleeping")
        time.sleep(2)
        j = get_outlier_event_display_logs()
        
    print(j)
    print("Outlier",j["data"]["is_outlier"]==[1])
    '''

    block9 = '''
    
    '''

    # run the code blocks inside a jupyter kernel
    from kale.utils.jupyter_utils import run_code as _kale_run_code
    from kale.utils.kfp_utils import \
        update_uimetadata as _kale_update_uimetadata
    blocks = (pipeline_parameters_block, data_loading_block,
              block1,
              block2,
              block3,
              block4,
              block5,
              block6,
              block7,
              block8,
              block9,
              )
    html_artifact = _kale_run_code(blocks)
    with open("/deploy_event_display.html", "w") as f:
        f.write(html_artifact)
    _kale_update_uimetadata('deploy_event_display')

    _kale_mlmd_utils.call("mark_execution_complete")


def explain(DEPLOY_PASSWORD: str, DEPLOY_SERVER: str, DEPLOY_USER: str, MINIO_ACCESS_KEY: str, MINIO_HOST: str, MINIO_SECRET_KEY: str):
    pipeline_parameters_block = '''
    DEPLOY_PASSWORD = "{}"
    DEPLOY_SERVER = "{}"
    DEPLOY_USER = "{}"
    MINIO_ACCESS_KEY = "{}"
    MINIO_HOST = "{}"
    MINIO_SECRET_KEY = "{}"
    '''.format(DEPLOY_PASSWORD, DEPLOY_SERVER, DEPLOY_USER, MINIO_ACCESS_KEY, MINIO_HOST, MINIO_SECRET_KEY)

    from kale.utils import mlmd_utils as _kale_mlmd_utils
    _kale_mlmd_utils.init_metadata()

    data_loading_block = '''
    # -----------------------DATA LOADING START--------------------------------
    from kale.marshal import utils as _kale_marshal_utils
    _kale_marshal_utils.set_kale_data_directory("/marshal")
    _kale_marshal_utils.set_kale_directory_file_names()
    X_test = _kale_marshal_utils.load("X_test")
    X_train = _kale_marshal_utils.load("X_train")
    adult = _kale_marshal_utils.load("adult")
    explainer = _kale_marshal_utils.load("explainer")
    model = _kale_marshal_utils.load("model")
    # -----------------------DATA LOADING END----------------------------------
    '''

    block1 = '''
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from alibi.explainers import AnchorTabular
    from alibi.datasets import fetch_adult
    from minio import Minio
    from minio.error import ResponseError
    from joblib import dump, load
    import dill
    import time
    import json
    from subprocess import run, Popen, PIPE
    from alibi_detect.utils.data import create_outlier_batch
    import swagger_client
    from swagger_client.rest import ApiException
    import yaml
    import json
    import urllib3
    urllib3.disable_warnings()
    '''

    block2 = '''
    def get_minio():
        return Minio(MINIO_HOST,
                        access_key=MINIO_ACCESS_KEY,
                        secret_key=MINIO_SECRET_KEY,
                        secure=False)
    '''

    block3 = '''
    def get_swagger_configuration():
        configuration = swagger_client.Configuration()
        configuration.host = 'http://seldon-deploy.seldon-system/seldon-deploy/api/v1alpha1'
        return configuration
    '''

    block4 = '''
    import requests

    from urllib.parse import urlparse

    KF_SESSION_COOKIE_NAME = "authservice_session"


    class SessionAuthenticator:
        """
        Returns the cookie token.
        """

        def __init__(self, server: str):
            self._server = server

            url = urlparse(server)
            self._host = f"{url.scheme}://{url.netloc}"

        def authenticate(self, user: str, password: str) -> str:
            auth_path = self._get_auth_path()
            success_path = self._submit_auth(auth_path, user, password)
            session_cookie = self._get_session_cookie(success_path)
            return session_cookie

        def _get_auth_path(self) -> str:
            # Send unauthenticated request
            res = requests.get(self._server, allow_redirects=False, verify=False)

            # Follow the 302 redirect
            oidc_path = res.headers["Location"]
            oidc_endpoint = f"{self._host}{oidc_path}"
            res = requests.get(oidc_endpoint, allow_redirects=False, verify=False)

            return res.headers["Location"]

        def _submit_auth(self, auth_path: str, user: str, password: str) -> str:
            auth_endpoint = f"{self._host}{auth_path}"
            auth_payload = {"login": user, "password": password}
            res = requests.post(auth_endpoint, auth_payload, allow_redirects=False, verify=False)
            
            login_path = res.headers["Location"]
            login_endpoint = f"{self._host}{login_path}"
            res = requests.get(login_endpoint, allow_redirects=False, verify=False)

            return res.headers["Location"]

        def _get_session_cookie(self, success_path: str) -> str:
            success_endpoint = f"{self._host}{success_path}"
            res = requests.get(success_endpoint, allow_redirects=False, verify=False)
            print(res.cookies)
            return res.cookies[KF_SESSION_COOKIE_NAME]

    def authenticate():
        authenticator = SessionAuthenticator(DEPLOY_SERVER)

        cookie = authenticator.authenticate(DEPLOY_USER, DEPLOY_PASSWORD)
        return cookie
    '''

    block5 = '''
    model.predict(X_train)
    idx = 0
    class_names = adult.target_names
    print('Prediction: ', class_names[explainer.predict_fn(X_test[idx].reshape(1, -1))[0]])
    '''

    block6 = '''
    explanation = explainer.explain(X_test[idx], threshold=0.95)
    print('Anchor: %s' % (' AND '.join(explanation['names'])))
    print('Precision: %.2f' % explanation['precision'])
    print('Coverage: %.2f' % explanation['coverage'])
    '''

    # run the code blocks inside a jupyter kernel
    from kale.utils.jupyter_utils import run_code as _kale_run_code
    from kale.utils.kfp_utils import \
        update_uimetadata as _kale_update_uimetadata
    blocks = (pipeline_parameters_block, data_loading_block,
              block1,
              block2,
              block3,
              block4,
              block5,
              block6,
              )
    html_artifact = _kale_run_code(blocks)
    with open("/explain.html", "w") as f:
        f.write(html_artifact)
    _kale_update_uimetadata('explain')

    _kale_mlmd_utils.call("mark_execution_complete")


setup_op = comp.func_to_container_op(
    setup, base_image='seldonio/jupyter-lab-alibi-kale:0.17')


build_model_op = comp.func_to_container_op(
    build_model, base_image='seldonio/jupyter-lab-alibi-kale:0.17')


build_outlier_op = comp.func_to_container_op(
    build_outlier, base_image='seldonio/jupyter-lab-alibi-kale:0.17')


train_explainer_op = comp.func_to_container_op(
    train_explainer, base_image='seldonio/jupyter-lab-alibi-kale:0.17')


deploy_seldon_op = comp.func_to_container_op(
    deploy_seldon, base_image='seldonio/jupyter-lab-alibi-kale:0.17')


deploy_outlier_op = comp.func_to_container_op(
    deploy_outlier, base_image='seldonio/jupyter-lab-alibi-kale:0.17')


deploy_event_display_op = comp.func_to_container_op(
    deploy_event_display, base_image='seldonio/jupyter-lab-alibi-kale:0.17')


explain_op = comp.func_to_container_op(
    explain, base_image='seldonio/jupyter-lab-alibi-kale:0.17')


@dsl.pipeline(
    name='seldon-e2e-adult-ih8de',
    description='Seldon e2e adult'
)
def auto_generated_pipeline(DEPLOY_NAMESPACE='admin', DEPLOY_PASSWORD='12341234', DEPLOY_SERVER='https://x.x.x.x/seldon-deploy/', DEPLOY_USER='admin@seldon.io', EXPLAINER_MODEL_PATH='sklearn/income/explainer', INCOME_MODEL_PATH='sklearn/income/model', MINIO_ACCESS_KEY='minio', MINIO_HOST='minio-service.kubeflow:9000', MINIO_MODEL_BUCKET='seldon', MINIO_SECRET_KEY='minio123', OUTLIER_MODEL_PATH='sklearn/income/outlier'):
    pvolumes_dict = OrderedDict()
    volume_step_names = []
    volume_name_parameters = []

    marshal_vop = dsl.VolumeOp(
        name="kale-marshal-volume",
        resource_name="kale-marshal-pvc",
        storage_class="nfs-client",
        modes=dsl.VOLUME_MODE_RWM,
        size="1Gi"
    )
    volume_step_names.append(marshal_vop.name)
    volume_name_parameters.append(marshal_vop.outputs["name"].full_name)
    pvolumes_dict['/marshal'] = marshal_vop.volume

    volume_step_names.sort()
    volume_name_parameters.sort()

    setup_task = setup_op(DEPLOY_PASSWORD, DEPLOY_SERVER, DEPLOY_USER, MINIO_ACCESS_KEY, MINIO_HOST, MINIO_MODEL_BUCKET, MINIO_SECRET_KEY)\
        .add_pvolumes(pvolumes_dict)\
        .after()
    setup_task.container.working_dir = "/home/jovyan"
    setup_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    output_artifacts = {}
    output_artifacts.update(
        {'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'})
    output_artifacts.update({'setup': '/setup.html'})
    setup_task.output_artifact_paths.update(output_artifacts)
    setup_task.add_pod_label("pipelines.kubeflow.org/metadata_written", "true")
    dep_names = setup_task.dependent_names + volume_step_names
    setup_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(dep_names))
    if volume_name_parameters:
        setup_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(volume_name_parameters))

    build_model_task = build_model_op(DEPLOY_PASSWORD, DEPLOY_SERVER, DEPLOY_USER, INCOME_MODEL_PATH, MINIO_ACCESS_KEY, MINIO_HOST, MINIO_MODEL_BUCKET, MINIO_SECRET_KEY)\
        .add_pvolumes(pvolumes_dict)\
        .after(setup_task)
    build_model_task.container.working_dir = "/home/jovyan"
    build_model_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    output_artifacts = {}
    output_artifacts.update(
        {'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'})
    output_artifacts.update({'build_model': '/build_model.html'})
    build_model_task.output_artifact_paths.update(output_artifacts)
    build_model_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    dep_names = build_model_task.dependent_names + volume_step_names
    build_model_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(dep_names))
    if volume_name_parameters:
        build_model_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(volume_name_parameters))

    build_outlier_task = build_outlier_op(DEPLOY_PASSWORD, DEPLOY_SERVER, DEPLOY_USER, MINIO_ACCESS_KEY, MINIO_HOST, MINIO_MODEL_BUCKET, MINIO_SECRET_KEY, OUTLIER_MODEL_PATH)\
        .add_pvolumes(pvolumes_dict)\
        .after(build_model_task)
    build_outlier_task.container.working_dir = "/home/jovyan"
    build_outlier_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    output_artifacts = {}
    output_artifacts.update(
        {'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'})
    output_artifacts.update({'build_outlier': '/build_outlier.html'})
    build_outlier_task.output_artifact_paths.update(output_artifacts)
    build_outlier_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    dep_names = build_outlier_task.dependent_names + volume_step_names
    build_outlier_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(dep_names))
    if volume_name_parameters:
        build_outlier_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(volume_name_parameters))

    train_explainer_task = train_explainer_op(DEPLOY_PASSWORD, DEPLOY_SERVER, DEPLOY_USER, EXPLAINER_MODEL_PATH, MINIO_ACCESS_KEY, MINIO_HOST, MINIO_MODEL_BUCKET, MINIO_SECRET_KEY)\
        .add_pvolumes(pvolumes_dict)\
        .after(build_model_task)
    train_explainer_task.container.working_dir = "/home/jovyan"
    train_explainer_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    output_artifacts = {}
    output_artifacts.update(
        {'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'})
    output_artifacts.update({'train_explainer': '/train_explainer.html'})
    train_explainer_task.output_artifact_paths.update(output_artifacts)
    train_explainer_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    dep_names = train_explainer_task.dependent_names + volume_step_names
    train_explainer_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(dep_names))
    if volume_name_parameters:
        train_explainer_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(volume_name_parameters))

    deploy_seldon_task = deploy_seldon_op(DEPLOY_NAMESPACE, DEPLOY_PASSWORD, DEPLOY_SERVER, DEPLOY_USER, EXPLAINER_MODEL_PATH, INCOME_MODEL_PATH, MINIO_ACCESS_KEY, MINIO_HOST, MINIO_MODEL_BUCKET, MINIO_SECRET_KEY)\
        .add_pvolumes(pvolumes_dict)\
        .after(train_explainer_task)
    deploy_seldon_task.container.working_dir = "/home/jovyan"
    deploy_seldon_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    output_artifacts = {}
    output_artifacts.update(
        {'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'})
    output_artifacts.update({'deploy_seldon': '/deploy_seldon.html'})
    deploy_seldon_task.output_artifact_paths.update(output_artifacts)
    deploy_seldon_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    dep_names = deploy_seldon_task.dependent_names + volume_step_names
    deploy_seldon_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(dep_names))
    if volume_name_parameters:
        deploy_seldon_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(volume_name_parameters))

    deploy_outlier_task = deploy_outlier_op(DEPLOY_PASSWORD, DEPLOY_SERVER, DEPLOY_USER, MINIO_ACCESS_KEY, MINIO_HOST, MINIO_SECRET_KEY)\
        .add_pvolumes(pvolumes_dict)\
        .after(deploy_seldon_task)
    deploy_outlier_task.container.working_dir = "/home/jovyan"
    deploy_outlier_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    output_artifacts = {}
    output_artifacts.update(
        {'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'})
    output_artifacts.update({'deploy_outlier': '/deploy_outlier.html'})
    deploy_outlier_task.output_artifact_paths.update(output_artifacts)
    deploy_outlier_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    dep_names = deploy_outlier_task.dependent_names + volume_step_names
    deploy_outlier_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(dep_names))
    if volume_name_parameters:
        deploy_outlier_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(volume_name_parameters))

    deploy_event_display_task = deploy_event_display_op(DEPLOY_NAMESPACE, DEPLOY_PASSWORD, DEPLOY_SERVER, DEPLOY_USER, MINIO_ACCESS_KEY, MINIO_HOST, MINIO_SECRET_KEY)\
        .add_pvolumes(pvolumes_dict)\
        .after(deploy_outlier_task)
    deploy_event_display_task.container.working_dir = "/home/jovyan"
    deploy_event_display_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    output_artifacts = {}
    output_artifacts.update(
        {'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'})
    output_artifacts.update(
        {'deploy_event_display': '/deploy_event_display.html'})
    deploy_event_display_task.output_artifact_paths.update(output_artifacts)
    deploy_event_display_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    dep_names = deploy_event_display_task.dependent_names + volume_step_names
    deploy_event_display_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(dep_names))
    if volume_name_parameters:
        deploy_event_display_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(volume_name_parameters))

    explain_task = explain_op(DEPLOY_PASSWORD, DEPLOY_SERVER, DEPLOY_USER, MINIO_ACCESS_KEY, MINIO_HOST, MINIO_SECRET_KEY)\
        .add_pvolumes(pvolumes_dict)\
        .after(train_explainer_task)
    explain_task.container.working_dir = "/home/jovyan"
    explain_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    output_artifacts = {}
    output_artifacts.update(
        {'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'})
    output_artifacts.update({'explain': '/explain.html'})
    explain_task.output_artifact_paths.update(output_artifacts)
    explain_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    dep_names = explain_task.dependent_names + volume_step_names
    explain_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(dep_names))
    if volume_name_parameters:
        explain_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(volume_name_parameters))


if __name__ == "__main__":
    pipeline_func = auto_generated_pipeline
    pipeline_filename = pipeline_func.__name__ + '.pipeline.tar.gz'
    import kfp.compiler as compiler
    compiler.Compiler().compile(pipeline_func, pipeline_filename)

    # Get or create an experiment and submit a pipeline run
    import kfp
    client = kfp.Client()
    experiment = client.create_experiment('seldon-e2e-adult')

    # Submit a pipeline run
    from kale.utils.kfp_utils import generate_run_name
    run_name = generate_run_name('seldon-e2e-adult-ih8de')
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, {})
