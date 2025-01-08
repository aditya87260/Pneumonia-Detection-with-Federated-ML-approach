import flwr as fl

# Custom aggregation example
class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        # Default FedAvg implementation
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_weights is not None:
            print(f"Round {rnd}: Aggregated global model weights from {len(results)} clients.")
        return aggregated_weights

# Start the server
if __name__ == "__main__":
    strategy = CustomFedAvg(
        fraction_fit=1.0,  # All clients participate
        min_fit_clients=2,
        min_available_clients=2,
    )
    fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=6),
    strategy=strategy,
    grpc_max_message_length= 1024 * 1024 * 1024

)

