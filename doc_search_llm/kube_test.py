from kubernetes import client, config
import os

def main():
    # Create a configuration object
    configuration = client.Configuration()

    # Specify the Kubernetes master IP
    host = os.getenv("HOST")
    configuration.host = host

    # Specify your token
    token = os.getenv("TOKEN")
    configuration.api_key["authorization"] = {token}
    configuration.api_key_prefix['authorization'] = 'Bearer'

    # Set the created configuration as default
    client.Configuration.set_default(configuration)

    v1 = client.CoreV1Api()

    print("Listing services with their info:\n")
    services = v1.list_namespaced_service(namespace="default")  # Change 'default' to your namespace
    for svc in services.items:
        print(f"Name: {svc.metadata.name}")
        print(f"Labels: {svc.metadata.labels}")
        print(f"Cluster IP: {svc.spec.cluster_ip}")
        print(f"Ports: {svc.spec.ports}\n")

if __name__ == '__main__':
    main()