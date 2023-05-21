from kubernetes import client, config
import os

def main():
    # Create a configuration object from the config file
    configuration = client.Configuration()

    # Specify the Kubernetes master IP
    host = os.getenv("HOST")
    configuration.host = host

    # Specify your token
    token = os.getenv("TOKEN")
    configuration.api_key["authorization"] = f'Bearer {token}'
    configuration.api_key_prefix['authorization'] = 'Bearer'

    configuration.verify_ssl = False

    # Set the created configuration as default
    #client.Configuration.set_default(configuration)
    client = client.ApiClient(configuration)

    v1 = client.CoreV1Api(client)

    print("Listing pods with their IPs:")
    ret = v1.list_pod_for_all_namespaces(watch=False)
    for i in ret.items:
        print("%s\t%s\t%s" %
              (i.status.pod_ip, i.metadata.namespace, i.metadata.name))

if __name__ == '__main__':
    main()