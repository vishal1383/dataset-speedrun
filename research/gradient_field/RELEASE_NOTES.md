Experimental compressed gradient representation: a 3,568,901-byte quantized teacher endpoint defines a differentiable loss over student parameters.

Across eight fresh initialization seeds and new audit documents, the potential reached mean loss 2.761729 after 200 updates.
The paired full-data control reached 2.996852 after 400 updates.
Mean student training time was 2.3345 seconds versus 13.8956 seconds on a DGX Spark.

The artifact encodes a trained checkpoint. Direct loading achieved loss 2.385663 in approximately 0.01076 seconds.
The source teacher required 1,600 updates and 45.48 seconds of training.
These measurements do not establish an end-to-end training speedup or a dataset-only compression record.

The release contains the exact frozen `q4.pt` artifact and its hash manifest.
Methods, all 18 development trials, sixteen confirmation runs, costs, and reproduction commands are under `research/gradient_field/`.
All confirmation runs used spark-a5a1 after the user instructed me to use only that Spark.

Source attribution: Ronen Eldan and Yuanzhi Li, TinyStories, https://huggingface.co/datasets/roneneldan/TinyStories.
Dataset-derived release assets use CDLA-Sharing-1.0: https://cdla.dev/sharing-1-0/.
Repository code uses MIT.
