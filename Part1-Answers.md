# Q1
`vocab_size` is 2, and it represents whether the pixel is black or white.
# Q2
Binarizing collapses every pixel to 0 or 1, so all the gray levels in between are lost: 200 and 255 both become 1, 10 and 120 both become 0. We keep "ink or not" and throw away intensity.

This is fine for MNIST, where a digit is defined by its shape, not its shading. It would hurt with color, where the meaning lives in the levels (skin tones, gradients) - a single threshold flattens all of that into black-and-white blobs.
# Q3
The model adds a positional embedding to each token, so it learns "what belongs at position `i` given positions 0..i-1". With row-major flattening, position `i` always maps to the same pixel (row `i // W`, col `i % W`), so position carries spatial meaning: position 0 is the top-left pixel, every image the same way.

That only works if the order is consistent across all images and the decode `.reshape(H, W)` uses the same row-major order. If the flatten order, the position indices, or the reshape disagree, position `i` no longer means a fixed location and the image comes out scrambled.
