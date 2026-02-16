"""
microgpt by @karpathy, the most atomic way to train and inference a GPT in pure, dependency-free Python,
translated into Julia.
"""

import Random
using Printf
Random.seed!(42)

# Let there be an input dataset `docs`: Vector{String} of documents (e.g. a dataset of names)
if !isfile("data/input.txt")
    import Downloads
    Downloads.download(
        "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt",
        "data/input.txt",
    )
end
docs = [strip(l) for l in split(strip(read("data/input.txt", String)), '\n') if !isempty(strip(l))]
Random.shuffle!(docs)
println("num docs: $(length(docs))")

# Let there be a Tokenizer to translate strings to discrete symbols and back
uchars = sort(collect(Set(join(docs))))  # unique characters become token ids 1..n (1-indexed)
BOS = length(uchars) + 1  # token id for the special Beginning of Sequence (BOS) token
vocab_size = length(uchars) + 1  # total number of unique tokens, +1 is for BOS
println("vocab size: $vocab_size")

# Let there be Autograd, to recursively apply the chain rule through a computation graph
mutable struct Value
    data::Float64                   # scalar value of this node calculated during forward pass
    grad::Float64                   # derivative of the loss w.r.t. this node, calculated in backward pass
    _children::Vector{Value}        # children of this node in the computation graph
    _local_grads::Vector{Float64}   # local derivative of this node w.r.t. its children
end

Value(data::Real) = Value(Float64(data), 0.0, Value[], Float64[])

Base.:+(a::Value, b::Value) = Value(a.data + b.data, 0.0, [a, b], [1.0, 1.0])
Base.:+(a::Value, b::Real)  = a + Value(b)
Base.:+(a::Real, b::Value)  = Value(a) + b

Base.:*(a::Value, b::Value) = Value(a.data * b.data, 0.0, [a, b], [b.data, a.data])
Base.:*(a::Value, b::Real)  = a * Value(b)
Base.:*(a::Real, b::Value)  = Value(a) * b

Base.:^(a::Value, b::Real)  = Value(a.data^b, 0.0, [a], [b * a.data^(b - 1)])
Base.inv(a::Value)          = Value(1.0 / a.data, 0.0, [a], [-1.0 / a.data^2])
Base.log(a::Value) = Value(log(a.data), 0.0, [a], [1.0 / a.data])
Base.exp(a::Value) = Value(exp(a.data), 0.0, [a], [exp(a.data)])
relu(a::Value)     = Value(max(0.0, a.data), 0.0, [a], [Float64(a.data > 0)])

Base.:-(a::Value)            = a * -1
Base.:-(a::Value, b::Value)  = a + (-b)
Base.:-(a::Value, b::Real)   = a + Value(-b)
Base.:-(a::Real, b::Value)   = Value(a) + (-b)
# Julia では b^(-1) のようなリテラル整数べき乗が literal_pow → inv に最適化されるため、inv が未定義でエラーになっています。
# Base.inv(a::Value) を追加しました。Julia は b^(-1) のようなリテラル整数べき乗を inv にディスパッチするため、^ の定義だけでは足りませんでした。
Base.:/(a::Value, b::Value)  = a * b^(-1)
Base.:/(a::Value, b::Real)   = a * (1.0 / b)
Base.:/(a::Real, b::Value)   = Value(a) * b^(-1)

Base.zero(::Type{Value}) = Value(0.0)

function backward!(loss::Value)
    topo = Value[]
    visited = Set{UInt}()
    function build_topo(v)
        id = objectid(v)
        if id ∉ visited
            push!(visited, id)
            for child in v._children
                build_topo(child)
            end
            push!(topo, v)
        end
    end
    build_topo(loss)
    loss.grad = 1.0
    for v in reverse(topo)
        for (child, lg) in zip(v._children, v._local_grads)
            child.grad += lg * v.grad
        end
    end
end

# Initialize the parameters, to store the knowledge of the model.
n_embd = 16      # embedding dimension
n_head = 4       # number of attention heads
n_layer = 1      # number of layers
block_size = 16  # maximum sequence length
head_dim = n_embd ÷ n_head  # dimension of each head

matrix(nout, nin; std=0.08) = [Value[Value(randn() * std) for _ in 1:nin] for _ in 1:nout]
state_dict = Dict{String, Vector{Vector{Value}}}(
    "wte"     => matrix(vocab_size, n_embd),
    "wpe"     => matrix(block_size, n_embd),
    "lm_head" => matrix(vocab_size, n_embd),
)
for i in 0:n_layer-1 # TODO 1:n_layer is better
    state_dict["layer$i.attn_wq"] = matrix(n_embd, n_embd)
    state_dict["layer$i.attn_wk"] = matrix(n_embd, n_embd)
    state_dict["layer$i.attn_wv"] = matrix(n_embd, n_embd)
    state_dict["layer$i.attn_wo"] = matrix(n_embd, n_embd)
    state_dict["layer$i.mlp_fc1"] = matrix(4 * n_embd, n_embd)
    state_dict["layer$i.mlp_fc2"] = matrix(n_embd, 4 * n_embd)
end
params = Value[p for mat in values(state_dict) for row in mat for p in row]
println("num params: $(length(params))")

# Define the model architecture: a stateless function mapping token sequence and parameters to logits.
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
function linear(x::Vector{Value}, w::Vector{Vector{Value}})
    [sum(wi * xi for (wi, xi) in zip(wo, x)) for wo in w]
end

function softmax(logits::Vector{Value})
    max_val = maximum(v.data for v in logits)
    exps = [exp(v - max_val) for v in logits]
    total = sum(exps)
    [e / total for e in exps]
end

function rmsnorm(x::Vector{Value})
    ms = sum(xi * xi for xi in x) / length(x)
    scale = (ms + 1e-5) ^ (-0.5)
    [xi * scale for xi in x]
end

function gpt(token_id, pos_id, keys, vals)
    tok_emb = state_dict["wte"][token_id]   # token embedding
    pos_emb = state_dict["wpe"][pos_id]     # position embedding
    x = [t + p for (t, p) in zip(tok_emb, pos_emb)]  # joint token and position embedding
    x = rmsnorm(x)

    for li in 0:n_layer-1
        # 1) Multi-head attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict["layer$li.attn_wq"])
        k = linear(x, state_dict["layer$li.attn_wk"])
        v = linear(x, state_dict["layer$li.attn_wv"])
        push!(keys[li+1], k)
        push!(vals[li+1], v)
        x_attn = Value[]
        for h in 1:n_head
            hr = (h-1)*head_dim+1 : h*head_dim
            q_h = q[hr]
            k_h = [ki[hr] for ki in keys[li+1]]
            v_h = [vi[hr] for vi in vals[li+1]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in 1:head_dim) / sqrt(Float64(head_dim))
                           for t in eachindex(k_h)]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in eachindex(v_h))
                        for j in 1:head_dim]
            append!(x_attn, head_out)
        end
        x = linear(x_attn, state_dict["layer$li.attn_wo"])
        x = [a + b for (a, b) in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict["layer$li.mlp_fc1"])
        x = [relu(xi) for xi in x]
        x = linear(x, state_dict["layer$li.mlp_fc2"])
        x = [a + b for (a, b) in zip(x, x_residual)]
    end

    linear(x, state_dict["lm_head"])
end

# Weighted random choice (dependency-free alternative to StatsBase.sample)
function weighted_choice(weights::Vector{Float64})
    r = rand() * sum(weights)
    cumul = 0.0
    for (i, w) in enumerate(weights)
        cumul += w
        if r <= cumul
            return i
        end
    end
    length(weights)
end

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m_buf = zeros(length(params))  # first moment buffer
v_buf = zeros(length(params))  # second moment buffer

# Repeat in sequence
num_steps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10_000

function run(num_steps)
    for step in 1:num_steps

        # Take single document, tokenize it, surround it with BOS special token on both sides
        doc = docs[mod1(step, length(docs))]
        tokens = [BOS; [findfirst(==(ch), uchars) for ch in doc]; BOS]
        n = min(block_size, length(tokens) - 1)
    
        # Forward the token sequence through the model, building up the computation graph all the way to the loss.
        keys = [Vector{Vector{Value}}() for _ in 1:n_layer]
        vals = [Vector{Vector{Value}}() for _ in 1:n_layer]
        losses = Value[]
        for pos_id in 1:n
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, vals)
            probs = softmax(logits)
            loss_t = -log(probs[target_id])
            push!(losses, loss_t)
        end
        loss = (1 / n) * sum(losses)  # final average loss over the document sequence. May yours be low.
    
        # Backward the loss, calculating the gradients with respect to all model parameters.
        backward!(loss)
    
        # Adam optimizer update: update the model parameters based on the corresponding gradients.
        lr_t = learning_rate * (1 - (step - 1) / num_steps)  # linear learning rate decay
        for (i, p) in enumerate(params)
            m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
            v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad^2
            m_hat = m_buf[i] / (1 - beta1^step)
            v_hat = v_buf[i] / (1 - beta2^step)
            p.data -= lr_t * m_hat / (sqrt(v_hat) + eps_adam)
            p.grad = 0.0
        end
        if step % (num_steps/50) == 0
            @printf("step %4d / %4d | loss %.4f\n", step, num_steps, loss.data)
        end 
    end
    
    # Inference: may the model babble back to us
    temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high
    println("\n--- inference (new, hallucinated names) ---")
    for sample_idx in 1:20
        keys = [Vector{Vector{Value}}() for _ in 1:n_layer]
        vals = [Vector{Vector{Value}}() for _ in 1:n_layer]
        token_id = BOS
        chars = Char[]
        for pos_id in 1:block_size
            logits = gpt(token_id, pos_id, keys, vals)
            probs = softmax([l / temperature for l in logits])
            token_id = weighted_choice([p.data for p in probs])
            if token_id == BOS
                break
            end
            push!(chars, uchars[token_id])
        end
        @printf("sample %2d: %s\n", sample_idx, String(chars))
    end
    
end
run(num_steps)