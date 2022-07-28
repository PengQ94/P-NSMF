from central_server import CentralServer
from third_party_server import ThirdPartyServer
from init_clients import init_clients
from train import train
from private_set_union import PSU
from communication_cost import communication_cost


if __name__ == "__main__":
    clients = init_clients()
    cenServer = CentralServer()
    thiParServer = ThirdPartyServer()

    PSU(cenServer, thiParServer, clients)
    train(cenServer, thiParServer, clients)
