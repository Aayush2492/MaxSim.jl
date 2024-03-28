module MaxSim

using Transformers
using Transformers.TextEncoders
using Transformers.HuggingFace
using Distances

textencoder, bert_model = hgf"bert-base-uncased"

function compute_max_sim(query, document)

    query_encoding = encode(textencoder, [[query]])
    query_embedding = bert_model(query_encoding).hidden_state
    query_embedding = query_embedding[:, :, 1]

    document_encoding = encode(textencoder, [[document]])
    document_embedding = bert_model(document_encoding).hidden_state
    document_embedding = document_embedding[:, :, 1]

    num_query_tokens = size(query_embedding, 2)
    num_document_tokens = size(document_embedding, 2)

    max_sim = 0
    for i in 1:num_query_tokens
        max_sim_per_query_token = -Inf
        for j in 1:num_document_tokens
            sim = 1 - cosine_dist(query_embedding[:, i], document_embedding[:, j])
            max_sim_per_query_token = max(max_sim_per_query_token, sim)
        end
        max_sim += max_sim_per_query_token
    end
    return max_sim
end

query = "What is the capital of France?"
document = "The capital of France is Paris."

max_sim = compute_max_sim(query, document)
println(max_sim)

end # module MaxSim