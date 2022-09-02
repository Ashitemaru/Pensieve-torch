import numpy as np

from config import config

# Helper variables
B_IN_MB = 1000000
BIT_IN_BYTE = 8
MILLISECOND_IN_SECOND = 1000

class ABREnvironment:
    def __init__(
        self,
        cooked_time_list,
        cooked_bandwidth_list,
        is_fixed,
        random_seed = config["random_seed"]
    ):
        assert len(cooked_time_list) == len(cooked_bandwidth_list)
        np.random.seed(random_seed)
        self.is_fixed = is_fixed

        self.cooked_time_list = cooked_time_list
        self.cooked_bandwidth_list = cooked_bandwidth_list

        self.video_chunk_counter = 0 # Index
        self.buffer_size = 0 # In ms

        # Pick a net trace
        if self.is_fixed:
            self.random_trace_id = 0
        else:
            self.random_trace_id = np.random.randint(len(self.cooked_time_list))
        self.cooked_time = self.cooked_time_list[self.random_trace_id] # In sec
        self.cooked_bandwidth = self.cooked_bandwidth_list[self.random_trace_id] # In M-bit per sec

        # Pick a start point of net trace
        if self.is_fixed:
            self.time_ptr = 1 # Index
        else:
            self.time_ptr = np.random.randint(1, len(self.cooked_time)) # Index
        self.last_mahimahi_time = self.cooked_time[self.time_ptr - 1] # In sec

        # Read in video size file
        self.video_size = {}
        for bitrate_level in range(len(config["video_bitrate"])):
            self.video_size[bitrate_level] = []
            with open(f"{config['video_size_dir']}/video_size_{bitrate_level}", "r") as handle:
                for line in handle:
                    self.video_size[bitrate_level].append(int(line.split()[0]))

    def get_video_chunk(self, bitrate_level):
        assert bitrate_level >= 0
        assert bitrate_level < len(config["video_bitrate"])

        video_chunk_size = self.video_size[bitrate_level][self.video_chunk_counter] # In byte
        delay = 0.0 # In ms
        byte_sent = 0.0 # In byte

        while True: # Try to download the chunk
            throughput = self.cooked_bandwidth[self.time_ptr] * B_IN_MB / BIT_IN_BYTE # In byte per sec
            duration = self.cooked_time[self.time_ptr] - self.last_mahimahi_time # In sec
            packet_payload = throughput * duration * config["packet_payload_portion"] # In byte

            if byte_sent + packet_payload > video_chunk_size: # Finish sending this chunk
                last_piece_time = (video_chunk_size - byte_sent) \
                    / throughput \
                    / config["packet_payload_portion"] # In sec
                delay += last_piece_time * MILLISECOND_IN_SECOND
                self.last_mahimahi_time += last_piece_time

                assert self.last_mahimahi_time <= self.cooked_time[self.time_ptr]
                break

            byte_sent += packet_payload
            delay += duration * MILLISECOND_IN_SECOND
            self.last_mahimahi_time = self.cooked_time[self.time_ptr]

            self.time_ptr += 1
            if self.time_ptr >= len(self.cooked_bandwidth): # When overflow, loop back
                self.time_ptr = 1
                self.last_mahimahi_time = 0

        delay += config["link_round_trip_time"] # Add RTT
        if not self.is_fixed:
            delay *= np.random.uniform(config["noise_lower_bound"], config["noise_upper_bound"]) # Add noise

        rebuf_time = np.maximum(delay - self.buffer_size, 0.0) # Calc re-buffer time

        # Update the buffer size
        # 1. When downloading the new chunk, drain the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)
        # 2. Fill in the new chunk
        self.buffer_size += config["video_chunk_len"]

        sleep_time = 0
        if self.buffer_size > config["buffer_threshold"]:
            exceeded_buffer_size = self.buffer_size - config["buffer_threshold"] # In ms

            # Drain the buffer until there exists enough space for the chunk
            # According to the paper, we will suspend downloading for 500 ms when the buffer is full
            sleep_time = np.ceil(exceeded_buffer_size / config["drain_buffer_sleep_time"]) \
                * config["drain_buffer_sleep_time"]
            self.buffer_size -= sleep_time

            while True: # Exhaust the sleep time
                duration = self.cooked_time[self.time_ptr] - self.last_mahimahi_time # In sec
                if duration > sleep_time / MILLISECOND_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECOND_IN_SECOND
                    break

                sleep_time -= duration * MILLISECOND_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.time_ptr]

                self.time_ptr += 1
                if self.time_ptr >= len(self.cooked_bandwidth): # When overflow, loop back
                    self.time_ptr = 1
                    self.last_mahimahi_time = 0

        self.video_chunk_counter += 1 # Move to the next chunk
        video_chunk_left = config["total_video_chunk"] - self.video_chunk_counter

        end_of_video = self.video_chunk_counter >= config["total_video_chunk"]
        if end_of_video:
            # Reset pointers
            self.buffer_size = 0
            self.video_chunk_counter = 0

            # Pick a new net trace
            if self.is_fixed:
                self.random_trace_id += 1
                if self.random_trace_id >= len(self.cooked_time_list):
                    self.random_trace_id = 0
            else:
                self.random_trace_id = np.random.randint(len(self.cooked_time_list))
            self.cooked_time = self.cooked_time_list[self.random_trace_id] # In sec
            self.cooked_bandwidth = self.cooked_bandwidth_list[self.random_trace_id] # In M-bit per sec

            # Pick a start point of net trace
            if self.is_fixed:
                self.time_ptr = 1 # Index
            else:
                self.time_ptr = np.random.randint(1, len(self.cooked_time)) # Index
            self.last_mahimahi_time = self.cooked_time[self.time_ptr - 1] # In sec

        next_chunk_size_list = []
        for i in range(len(config["video_bitrate"])):
            next_chunk_size_list.append(self.video_size[i][self.video_chunk_counter])

        return (
            delay,
            sleep_time,
            self.buffer_size / MILLISECOND_IN_SECOND,
            rebuf_time / MILLISECOND_IN_SECOND,
            video_chunk_size,
            next_chunk_size_list,
            end_of_video,
            video_chunk_left,
        )

    def hard_reset(self, trace_id, time_ptr):
        # Hard reset to a new trance, reset all the pointers
        self.random_trace_id = trace_id
        self.buffer_size = 0
        self.video_chunk_counter = 0
        self.cooked_time = self.cooked_time_list[self.random_trace_id]
        self.cooked_bandwidth = self.cooked_bandwidth_list[self.random_trace_id]

        # Hard reset to a new start point of net trace
        self.time_ptr = time_ptr % len(self.cooked_time) # Index
        self.last_mahimahi_time = self.cooked_time[self.time_ptr - 1] # In sec

if __name__ == "__main__":
    pass