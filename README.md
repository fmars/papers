
## Scaling Law for Neural Network Models
10/23

* [https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)
* The loss scales as a power-law with model size, dataset size, and the amount of compute used for training
* Constant values in power law equations derived from empirical data
* Equations of num parameters and flops per token in a Transformer model
* Insightful analysis on equations related to overfitting and efficient batch size

# MoE 
10/23

OpenAI techniques for training large neural networks

* [https://openai.com/research/techniques-for-training-large-neural-networks](https://openai.com/research/techniques-for-training-large-neural-networks) 


## Sparsely-Gated MoE 



* [https://arxiv.org/abs/1701.06538](https://arxiv.org/abs/1701.06538) 
* MoE has been created 20 years ago: y = sum(G(x) * E(x))
* Gating network: 
    * 1) sparsity: only top-k experts will be computed, 
    * 2) a noise on the input x for Gating computation
* Addressing Issues
    * Shrinking batch problem: data parallel standard layers but model parallel experts (only one replica)
    * Network bandwidth: increase n_layer and hidden dimension in experts
    * Balancing Expert utilization: introduce additional loss function to encourage equal importance
* Q: How is routing logic actually implemented? 


## Switch Transformers



* [https://arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961) 
* Limitation of Sparsely-Gated MoE: computation efficiency, complexity, instability
* Switch Transformer
    * Parameter count, independent of total computation performed, is a separately important axis on which to scale
    * Simplify expert routing: instead of top-k, make k=1 in routing
    * Efficient sparse routing: 
        * Set expert capacity, and skip computation for tokens that exceeds expert capacity
        * Single auxiliary loss function to balance expert load and importance
* Solve instability: 1) selective precision, 2) parameter initialization, 3) regularization
* Advanced results: 
    * 1) advanced performance on time and step basis analysis, 
    * 2) advanced performance on fine tune, distillation, etc
* Perf analysis (compute & comm) over data, model, and expert parallel 


## GShard 



* [https://arxiv.org/abs/2006.16668](https://arxiv.org/abs/2006.16668) 
* A easy to use library that partitions gated MoE model with high efficiency
* Design principles: 
    * 1) use gated MoE to achieve sub-linear scaling between computation/communication requirements and model capacity
    * 2) separate model arch code with infra (partitioning and perf optimization) code
    * 3) still use SPMD to retain compiler scalability
* Model: 1) top-k is independent of num of experts, 2) shard MoE across devices but replicate other layers
* Parallelization: 
    * partition input tokens into groups, each of which runs gating and routing in parallel
    * einsum to denote the algorithm
    * use annotation to denote sharding (dimension, shard) and replication info
* XLA compiler partitioner implementation details: communication primitives, operators
* Experiment on Gated MoE, with perf (memory, compute, communication) analysis and best practice

# LoRA 

09/23

## LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS
Terms
- The rank of a matrix
- autoregressive model

Existing transfer learning strategies
- Section 6, a survey of well known transfer learning strategies
- Naive fine tune
  - Substantial computation requirements
  - Overfitting due to small size of domain specific data
- Adding adapter layer
  - Extra latency
  - Optimizing input layer transformation (TODO what does this do)]

Strategy
- Over-parameterized matrix delta W has a lower intrinsic dimension
- Use B[d*r] @ A[r*k] to simulate delta W[d*k]


## Pathway: Asynchronous distributed dataflow for ML

Context
- Unlike SPMD in pytorch (all GPUs execute the same program in a synchronous manner), TF uses MPMD in an asynchronous manner with a controller to decide which TPU runs what on which TPUs

Problem statement 
- In TF one training program can only run within a Pod (usually with a few thousand TPU cores). No existing way to run across Pods, which are connected through Datacenter network
Native dataflow in asynchronous execution is inefficient

Solution
- Parallel dispatch: controller has the global information thus can precompute / compile tensor shape ahead of execution, which is used for tensor allocation and RDMA across TPUs
- Some scheduling/data management implementations to avoid deadlock, optimize cross Pod execution

## GPipe: Pipeline Parallelism

Problem statement
- Single GPU has memory limitation, cannot fit larger models
- Basic pipeline parallelism faces memory inefficiencies

Solution
- To solve memory limitation of a single GPU, shard model on layer dimension and allocate to different GPUs
- To reduce inefficiency caused by pipeline bubble, introduce microbatch
- To further reduce memory consumption, implement re-materialization (i.e. activation checkpointing)

Perf Analysis
- Bubble time:O(K-1M+K-1), K is num GPUs, M is num of microbatch, K is num of GPUs 
  - In our setting, it’s O(pipe_depth-1max_n_microbatches_per_group-pipe_depth-1)
  - If we ensure max_n_microbatches_per_group=1, M becomes to  global_batch_size/microbatch_size
  - Observation from paper: bubble time is negligible when M>=4K
- Peak memory requirements per GPU
  - Vanilla version: O(d_model *global_batch_size *n_layer )
  - Pipeline: O( d_model *global_batch_size *n_layerpipe_depth )
  - Pipeline + checkpointing:  O(d_model*global_batch_size*(1+n_layerpipe_depth * n_microbatch) )
- Effective computation after activation checkpointing
  - 65% is effective computation
  - 22% is re-compute forward pass for activation checkpointing
  - 10% roughly evenly distributed across 1) load balance, 2) bubble overhead, 3) others
- Overall, GPipe is able to 
  - partition the model memory footprint with the cost of computation inefficiency
  - reduce peak memory requirement with the cost of extra forward pass computation

Further reading
- A more sophisticated pipeline implementation, PipeDream by MSFT
- https://arxiv.org/abs/1806.03377 

## Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism 

Problem Statement
- 1) large model size cannot fit into one gpu memory, 2) efficiently scale computation to multiple GPUs
- Data parallel: by increasing batch size proportionally to worker size, one observed that nearly linear scaling in throughput. However, larger global batch size resulted into ML issue (e.g. accuracy, longer time convergence)
- Model parallel: 
  - parameter server style solutions face consistency issue
  - Gpipe related solutions requires dedicated framework/compiler/etc

Solution
- Tensor parallel, dedicated for transformer, easy to implement with a few changes on pytorch
- Partition 2 layered MLP by column wise followed by row wise -> 2 synchronization needed 
- Partition self-attention by splitting Wq,Wk,Wv by attention head
- Partition embedding table similarly

Perf Analysis: Megatron vs Data Parallel
- Data parallel faces problem due to increasing number of global batch size, which resulting into ML issues
(TODO) Different requirements on communication
- Megatron: O(batch_size * n_ctx * 
- Data parallel: O(
- In Megatron, communication and computation have to happen sequentially, whereas in data parallel they can overlap with each other
- Megatron also requires input X to be copies across GPUs

Perf Analysis: Metatron vs GPipe
- Metatron is specialized for transformer architecture
- [TODO] GPipe has to use activation checkpointing thus causing redundant computation. This is because to maintain reasonable GPU efficiency, it needs to have a big enough number for microbatch size. Meanwhile, in order to reduce bubbles, it needs to have a big enough number of microbatch. So it has at least a big number for memory.
- [TODO] GPipe has less network communication requirements as the comm is proportional to n_gpu rather than n_layers

Metatron-LM
- Due to Megatron’s strict requirements on network (both throughput and latency), it doesn’t scale well beyond 8 GPUs, otherwise inter-node network is required: i.e. n_op_shard <=8
- n_op_shard can divide n_heads and emb etc


## ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

Problem statement
- DP requires the model to fit into a single gpu. Also larger batch size cause slower convergence
- MP (Megatron) doesn’t scale beyond a node
- All other solutions introduce a significant amount of redundant memory

  
Context: model memory footprint analysis 
- Model state: optimizer state, parameter, gradient = (2+2+12)*X
- Residual state: activation (proportional to batch_size, n_ctx, and d_model), buffer, fragmentation

Solution
- ZeRO-DP: partition param, grad, and opt. Copy param before forward. Each shard only need to store and update corresponding grad and opt
- ZeRO-R: some engineering things

Perf Analysis
- Unlike Megatron, which split the operation, where each shard computes the part of computation, thus requires to all_reduce the results of each step (e.g. self attention, mlp). Since it’s intermediate computation data, the size is proportional to batch_size, n_ctx, etc. Thus very large. 
- Besides, since all the shards compute on the same input X, it need to copy X to all shards
- ZeRO copies the params to all shards. And only synchronize on gradient and weights, which is only proportional to model param size, which is fixed and relatively small
- Also it doesn’t need to copy input X, since each shard works on its own input, and its own forward and backward passes. It’s more of a upgraded version of DP

## Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM

## Scaling Distributed Machine Learning with the Parameter Server
- A general purpose ML system capable of scaling to industry scale
- Architecture: parameter server, worker, resource manager
- Consistency: sequential, eventual, bounded delay
- Fault tolerance: consistent hashing with replication
- Network optimization: compression, caching, filter
- Versioning: vector clock to support versioning with range query



# Distributed System

- 04/23

1. [Kafka: a Distributed Messaging System for Log Processing](https://notes.stephenholiday.com/Kafka.pdf)
2. [ZooKeeper: Wait-free coordination for Internet-scale systems](https://www.usenix.org/legacy/event/atc10/tech/full_papers/Hunt.pdf)
