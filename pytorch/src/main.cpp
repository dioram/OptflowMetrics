#include <torch/torch.h>
#include <torch/script.h>

int main(int argc, char* argv[]) {
	auto model = torch::jit::load("D:/Work/RAFT/raft.pt");
	return 0;
}