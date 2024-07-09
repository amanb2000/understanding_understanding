import torch
import torch.nn.functional as F

def calculate_purity(tensors, class_ids):
    """
    tensors: [batch, num_tokens, d_model] -- a given layer's internal reps
    class_ids: tensor of shape [batch] -- the answers for each problem
    """
    # Flatten the tensors along the num_tokens dimension
    tensors = tensors.view(tensors.size(0), -1)  # Shape: (Batch, num_tokens * d_model)

    # Normalize the vectors (ensuring we're working with unit vectors)
    tensors = F.normalize(tensors, p=2, dim=1)

    # Compute the cosine similarity matrix
    similarity_matrix = torch.mm(tensors, tensors.t())  # Shape: (Batch, Batch)

    # Transform similarity matrix from [-1, 1] to [0, 2]
    similarity_matrix = similarity_matrix + 1

    # Mask to exclude self-comparisons
    eye_mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=similarity_matrix.device)

    # Initialize masks for intra-class and inter-class similarities
    intra_class_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
    inter_class_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)

    for i in range(len(class_ids)):
        for j in range(len(class_ids)):
            if class_ids[i] == class_ids[j]:
                intra_class_mask[i, j] = True
            else:
                inter_class_mask[i, j] = True

    # Exclude diagonal (self-similarities)
    intra_class_mask[eye_mask] = False

    # Calculate average similarities
    if intra_class_mask.any():
        intra_class_similarity = similarity_matrix[intra_class_mask].mean()
    else:
        intra_class_similarity = torch.tensor(0.0)

    if inter_class_mask.any():
        inter_class_similarity = similarity_matrix[inter_class_mask].mean()
    else:
        inter_class_similarity = torch.tensor(0.0)

    # Handle cases where inter-class similarity is zero to avoid division by zero
    if inter_class_similarity == 0:
        purity_score = torch.tensor(float('inf'))  # Consider infinite purity if there is no inter-class similarity
    else:
        purity_score = intra_class_similarity / inter_class_similarity

    return purity_score.item()

# # Example usage
# batch_size = 10
# num_tokens = 5
# d_model = 3

# # Random tensors and class_ids for demonstration
# tensors = torch.randn(batch_size, num_tokens, d_model)
# class_ids = torch.tensor([1, 2, 1, 2, 1, 1, 2, 2, 1, 1])

# # skew the tensors to have similar vectors for class 1 and class
# tensors[class_ids == 1] = tensors[class_ids == 1] + 10
# tensors[class_ids == 2] = tensors[class_ids == 2] - 10


# purity = calculate_purity(tensors, class_ids)
# print("Purity score:", purity)
