import time
import random

class FiveGNetworkSimulator:
    """
    Simulates 5G network transmission with bandwidth, latency, and noise factors.
    """

    def __init__(self, bandwidth=1000, latency_ms=5, packet_loss_rate=0.02):
        """
        Args:
            bandwidth (float): Bandwidth in Mbps (e.g., 1000 Mbps for 5G).
            latency_ms (float): Base latency in milliseconds.
            packet_loss_rate (float): Probability (0–1) of packet loss.
        """
        self.bandwidth = bandwidth
        self.latency_ms = latency_ms
        self.packet_loss_rate = packet_loss_rate

    def transmit(self, data, filename):
        """
        Simulates transmitting data over a 5G link.

        Args:
            data (bytes): The data to transmit.
            filename (str): Name of the file being transmitted.

        Returns:
            dict: Stats including delay, loss, and signal quality.
        """
        data_size_kb = len(data) / 1024
        base_time = data_size_kb / (self.bandwidth * 125)  # 1 Mbps = 125 KB/s
        jitter = random.uniform(0.0005, 0.005)  # micro-fluctuations
        network_delay = base_time + (self.latency_ms / 1000.0) + jitter
        time.sleep(network_delay)

        # Simulate signal fluctuations (in dB)
        signal_quality_db = random.uniform(-85, -45)

        # Simulate random packet loss
        loss_event = random.random() < self.packet_loss_rate

        # Return all transmission statistics
        return {
            "filename": filename,
            "data_size_kb": data_size_kb,
            "delay_s": network_delay,
            "lost": loss_event,
            "signal_quality_db": signal_quality_db
        }
