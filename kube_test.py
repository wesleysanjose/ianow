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
            log.info(f"Namespace: {pod.metadata.namespace}")
            log.info(f"Name: {pod.metadata.name}")
            log.info(f"Status: {pod.status.phase}")
            log.info(f"IP: {pod.status.pod_ip}")
            log.info(f"Node: {pod.spec.node_name}")
            log.info(f"Labels: {pod.metadata.labels}")
            log.info(f"Annotations: {pod.metadata.annotations}")
            log.info(f"Containers: {pod.spec.containers}")
            log.info(f"Volumes: {pod.spec.volumes}")
            log.info(f"Host IP: {pod.status.host_ip}")
            log.info(f"QoS Class: {pod.status.qos_class}")
            log.info(f"Service Account: {pod.spec.service_account}")
            log.info(f"Service Account Name: {pod.spec.service_account_name}")
            log.info(f"Node Selector: {pod.spec.node_selector}")
            log.info(f"Node Name: {pod.spec.node_name}")
            log.info(f"Restart Policy: {pod.spec.restart_policy}")
            log.info(f"DNS Policy: {pod.spec.dns_policy}")
            log.info(f"Scheduler Name: {pod.spec.scheduler_name}")
            log.info(f"Security Context: {pod.spec.security_context}")
            log.info(f"Termination Grace Period Seconds: {pod.spec.termination_grace_period_seconds}")
            log.info(f"Image Pull Secrets: {pod.spec.image_pull_secrets}")
            log.info(f"Host Aliases: {pod.spec.host_aliases}")
            log.info(f"Priority Class Name: {pod.spec.priority_class_name}")
            log.info(f"Priority: {pod.spec.priority}")
            log.info(f"DNS Config: {pod.spec.dns_config}")
            log.info(f"Readiness Gates: {pod.spec.readiness_gates}")
            log.info(f"Runtime Class Name: {pod.spec.runtime_class_name}")
            log.info(f"Enable Service Links: {pod.spec.enable_service_links}")
            log.info(f"Preemption Policy: {pod.spec.preemption_policy}")
            log.info(f"Overhead: {pod.spec.overhead}")
            log.info(f"Topology Spread Constraints: {pod.spec.topology_spread_constraints}")
            log.info(f"Affinity: {pod.spec.affinity}")
            log.info(f"Scheduler Name: {pod.spec.scheduler_name}")
            log.info(f"DNS Policy: {pod.spec.dns_policy}")
        # dump_json(pod_list.items)
    except Exception as e:
        log.error(f'Error listing pods: {e}')
        raise e

def main():
    args = load_parser()

    v1 = config_from_file(args=args)

    get_all_pods(v1=v1)

if __name__ == '__main__':
    main()