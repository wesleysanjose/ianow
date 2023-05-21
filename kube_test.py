from kubernetes import client, config
import os

from utils.simple_logger import Log
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
log = Log.get_logger(__name__)

def main():
    # Create a configuration object from the config file
    configuration = client.Configuration()

    # Specify the Kubernetes master IP
    host = os.getenv("HOST")
    log.debug(f'host: {host}')
    configuration.host = host

    
    token = os.getenv("TOKEN")
    # Specify your token last 4 chars
    log.debug(f'token ends with: {token[-4:]}')

    configuration.api_key = {"authorization": "Bearer " + token}

    configuration.verify_ssl = False

    # Set the created configuration as default
    #client.Configuration.set_default(configuration)
    api_client = client.ApiClient(configuration)

    v1 = client.CoreV1Api(api_client)

    print("Listing pods with their IPs:")
    ret = v1.list_pod_for_all_namespaces(watch=False)
    for i in ret.items:
        #print("%s\t%s\t%s" %
        #      (i.status.pod_ip, i.metadata.namespace, i.metadata.name))
        print(i)
if __name__ == '__main__':
    main()