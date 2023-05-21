import argparse
from kubernetes import client, config
import kubernetes
import os

import yaml

from flatten_json import flatten
import json

from utils.simple_logger import Log
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
log = Log.get_logger(__name__)

def make_k8s_client(kubeconfig: dict) -> kubernetes.client.CoreV1Api:
    api_client = kubernetes.config.new_client_from_config_dict(kubeconfig)
    return kubernetes.client.CoreV1Api(api_client)

def config_from_file(args):
    if args.config_path is not None:
        kube_config = args.config_path
    else:
        kube_config = 'client.config'
    try:
        with open(kube_config) as f:
            kube_config = yaml.safe_load(f)

        v1 = make_k8s_client(kube_config)
        return v1
    except Exception as e:
        log.error(f'Error loading kubeconfig: {e}')
        raise e
    
def config_from_env():
    try:
        # Create a configuration object from the config file
        configuration = client.Configuration()

        # Specify the Kubernetes master IP
        host = os.getenv("KUBERNETES_SERVICE_ENDPOINT")
        log.debug(f'host: {host}')
        configuration.host = host

        
        token = os.getenv("TOKEN")
        # Specify your token last 4 chars
        if token is not None:
            log.debug(f'token ends with: {token[-4:]}')
        else:
            log.debug(f'token is None')

        configuration.api_key = {"authorization": "Bearer " + token}

        # configuration.verify_ssl = False

        # Set the created configuration as default
        api_client = client.ApiClient(configuration)
        v1 = client.CoreV1Api(api_client)
        return v1
    except Exception as e:
        log.error(f'Error loading kubeconfig: {e}')
        raise e

def load_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config_path', type=str,
                        help='config path for Kubernetes api client')
    args = parser.parse_args()
    log.debug(f'args: {args}')
    return args

def dump_json(data):
    try:
        with open("output.json","w") as write_file:
            json.dump([d for d in data], write_file)
            write_file.close()
        
        flattened = [flatten(d) for d in data]
        with open("output_flattened.json","w") as write_file:
            json.dump(flattened, write_file)
            write_file.close()
    except Exception as e:
        log.error(f'Error dumping json: {e}')
        raise e

def get_all_pods(v1):
    try:
        pod_list = v1.list_pod_for_all_namespaces(watch=False)
        for pod in pod_list.items:
            log.info(f"Namespace: {pod.metadata.namespace}, Pod: {pod.metadata.name}")
        
        dump_json(pod_list.items)
    except Exception as e:
        log.error(f'Error listing pods: {e}')
        raise e

def main():
    args = load_parser()

    v1 = config_from_file(args=args)

    get_all_pods(v1=v1)

if __name__ == '__main__':
    main()