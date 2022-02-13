# The following uses locality-sensitive-min-hashing to efficiently represent the underlying dataset
# for Jaccard distance calculation.

import random
import sys

b = 40                                                  # Number of bands
r = 24                                                  # Number of rows per band
T = 0.85                                                # Similarity threshold
N_SHINGLES = 8193                                       # Total number of possible shingles
A = [random.randint(1, b * r) for _ in xrange(b * r)]   # a's for permutations, Uniform{1,...,N}
B = [random.randint(0, b * r) for _ in xrange(b * r)]   # b's for permutations, Uniform{0,...,N}
p = 15486173                                            # Large prime

m = 10000000                                            # Number of bins in the hash table
p_1 = 179424799                                         # Another large prime
A_1 = [random.randint(1, p_1 - 1) for _ in xrange(r)]   # a's for hashing bands, Uniform{1,...,p_1-1}
B_1 = [random.randint(0, p_1 - 1) for _ in xrange(r)]   # b's for hashing bands, Uniform{0,...,p_1-1}

K = 1000                                                # Constant for computing keys in the mapper (K < b)

# Mapper computes hash values for bands of the signature of the given document
# Yields <hash value + (band number) / K, document>
def mapper(key, value):
    # key: None
    # value: space-separated list of shingles present in the document (preceded by the document ID)
    shingles = value.strip("\n").split(" ") # Split the line on spaces
    pagename = shingles[0]                  # The name of the page
    pageID = int(pagename.split("_")[1])    # The ID of the page
    shingles = shingles[1:]                 # Set of shingles that are present in the page
    shingleVector = [0] * N_SHINGLES        # Represents the page as a vector of shingles
    # Construct the vector of the page
    for sh in shingles:
        shingleVector[int(sh)] = 1          # Assign 1 to entries corresponding to the shingles that are present
    signature = [sys.maxint] * (b * r)      # The signature of the document, instantiate all entries to inf
    #######################
    # Computing signature #
    #######################
    for i in range(0, N_SHINGLES):
        if shingleVector[i] == 1:
            for j in range(0, b * r):
                pi = ((A[j] * i + B[j]) % p) % (b * r)  # Permuted index
                signature[j] = min([signature[j], pi])  # Update the signature, if needed
    #################
    # Hashing bands #
    #################
    for band in range(0, b):
        s = signature[0 + r * band : r * (band + 1)]    # Current band
        hs = 0                                          # Hash function value
        for i in range(0, r):
            hs += ((A_1[i] * s[i] + B_1[i]) % p_1) % m
        hs = hs % m
        hs = float(hs) + (float(band) / float(K))       # Add [(band number) / K] to avoid collisions from different bands

        # Emit the modified hash value and the document shingles
        yield float(hs), value



# Reducer should output a key, value pair of two integers representing the ID's of duplicated pages
# the smaller ID being the key and the larger the value.
def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Iterate through all pairs
    for i in range(0, len(values)):
        for j in range(i + 1, len(values)):
            ######################
            # Jaccard similarity #
            ######################
            # Construct shingle vectors for documents i and j
            line_i = values[i]
            line_j = values[j]
            shingles_i = line_i.strip("\n").split(" ")[1:]
            shingles_j = line_j.strip("\n").split(" ")[1:]
            shingleVector_i = [0] * N_SHINGLES
            shingleVector_j = [0] * N_SHINGLES
            for sh in shingles_i:
                shingleVector_i[int(sh)] = 1
            for sh in shingles_j:
                shingleVector_j[int(sh)] = 1
            # Calculate the cardinality of the intersection
            intersec = 0
            for el in range(0, N_SHINGLES):
                if shingleVector_i[el] == 1 and shingleVector_j[el] == 1:
                    intersec = intersec + 1
            # Calculate the cardinality of the union
            uni = sum(shingleVector_i) + sum(shingleVector_j) - intersec
            sim = float(intersec) / float(uni)                  # Jaccard similarity
            # If similarity > T, then return the near duplicate pair
            if (sim >= T):
                pagename_i = line_i.strip("\n").split(" ")[0]   # The name of page i
                pageID_i = int(pagename_i.split("_")[1])        # The ID of page i
                pagename_j = line_j.strip("\n").split(" ")[0]   # The name of page j
                pageID_j = int(pagename_j.split("_")[1])        # The ID of page j
                # Emit the pair of ID's
                yield min([pageID_i, pageID_j]), max([pageID_i, pageID_j])
