module MaxSim

using Transformers
using Transformers.TextEncoders
using Transformers.HuggingFace
using LinearAlgebra

textencoder, bert_model = hgf"bert-base-uncased"

function compute_max_sim(query, document)
    """
    Compute the maximum similarity between a query and a document.
    Complexity: O(n * m), where n is the number of tokens in the query and m is the number of tokens in the document.

    # Arguments
    - `query`: The query string.
    - `document`: The document string.

    # Returns
    The maximum similarity score between the query and document.
    
    """

    query_encoding = encode(textencoder, [[query]])
    query_embedding = bert_model(query_encoding).hidden_state
    query_embedding = query_embedding[:, :, 1]

    document_encoding = encode(textencoder, [[document]])
    document_embedding = bert_model(document_encoding).hidden_state
    document_embedding = document_embedding[:, :, 1]
    
    num_query_tokens = size(query_embedding, 2)
    num_document_tokens = size(document_embedding, 2)

    # Normalize the embeddings
    for i in 1:num_query_tokens
        query_embedding[:, i] = query_embedding[:, i] / norm(query_embedding[:, i])
    end
    for i in 1:num_document_tokens
        document_embedding[:, i] = document_embedding[:, i] / norm(document_embedding[:, i])
    end

    similarity_score_matrix = zeros(Float64, num_query_tokens, num_document_tokens)
    mul!(similarity_score_matrix, query_embedding', document_embedding)

    max_sim = 0
    for i in 1:num_query_tokens
        max_sim_per_query_token = -Inf
        for j in 1:num_document_tokens
            sim = similarity_score_matrix[i, j]
            max_sim_per_query_token = max(max_sim_per_query_token, sim)
        end
        max_sim += max_sim_per_query_token
    end
    return max_sim
end

# A long query and document to test the function
query = "Which planet in the solar system of the Milky Way galaxy is known for its rings?"
document = "Saturn is the sixth planet from the Sun and the second-largest in the Solar System, after Jupiter. It is a gas giant with an average radius of about nine times that of Earth. It only has one-eighth the average density of Earth; however, with its larger volume, Saturn is over 95 times more massive. Saturn is named after the Roman god of wealth and agriculture. Its astronomical symbol (♄) represents the god's sickle."

max_sim = compute_max_sim(query, document)
println(max_sim)

end # module MaxSim