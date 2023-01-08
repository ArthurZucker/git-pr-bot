# git-pr-bot (example with the accelerate library)

# Inputs :

A raw git diff file like the following : `https://github.com/huggingface/accelerate/pull/1.diff`
```diff
diff --git a/.gitignore b/.gitignore
index b6e47617d..3b89c957d 100644
--- a/.gitignore
+++ b/.gitignore
@@ -127,3 +127,6 @@ dmypy.json
 
 # Pyre type checker
 .pyre/
+
+# VSCode
+.vscode
diff --git a/README.md b/README.md
index 781eea3c9..fcc20a12a 100644
--- a/README.md
+++ b/README.md
@@ -34,7 +34,98 @@
 <p>Run your *raw* PyTorch training script on any kind of device
 </h3>
 
-🤗 Accelerate provides an easy API to make your scripts run with mixed precision and on any kind of distributed setting (multi-GPUs, TPUs etc.) while still letting you write your own training loop. The same code can then runs seamlessly on your local machine for debugging or your training environment.
+🤗 Accelerate was created for PyTorch users who like to write the training loop of PyTorch models but are reluctant to write and maintain the boiler code needed to use multi-GPUs/TPU/fp16.
+
+🤗 Accelerate abstracts exactly and only the boiler code related to multi-GPUs/TPU/fp16 and let the rest of your code unchanged.
+
+Here is an example:
+
+<table>
+<tr>
+<th> Original training code (CPU or mono-GPU only)</th>
+<th> With Accelerate for CPU/GPU/multi-GPUs/TPUs/fp16 </th>
+</tr>
+<tr>
+<td>
+
+```python
+import torch
+import torch.nn.functional as F
+from datasets import load_dataset
+
+
+
+device = 'cpu'
+
+model = torch.nn.Transformer().to(device)
+optim = torch.optim.Adam(model.parameters())
+
+dataset = load_dataset('my_dataset')
+data = torch.utils.data.Dataloader(dataset)
+
+
+
+
+model.train()
+for epoch in range(10):
+    for source, targets in data:
+        source = source.to(device)
+        targets = targets.to(device)
+
+        optimizer.zero_grad()
+
+        output = model(source, targets)
+        loss = F.cross_entropy(output, targets)
+
+        loss.backward()
+
+        optimizer.step()
+```
+
+</td>
+<td>
+
+```python
+  import torch
+  import torch.nn.functional as F
+  from datasets import load_dataset
+
++ from accelerate import Accelerator
++ accelerator = Accelerator()
++ device = accelerator.device
+
+  model = torch.nn.Transformer().to(device)
+  optim = torch.optim.Adam(model.parameters())
+
+  dataset = load_dataset('my_dataset')
+  data = torch.utils.data.Dataloader(dataset)
+
++ model, optim, data = accelerator.prepare(
+                            model, optim, data)
+
+  model.train()
+  for epoch in range(10):
+      for source, targets in data:
+          source = source.to(device)
+          targets = targets.to(device)
+
+          optimizer.zero_grad()
+
+          output = model(source, targets)
+          loss = F.cross_entropy(output, targets)
+
++         accelerate.backward(loss)
+
+          optimizer.step()
+```
+
+</td>
+</tr>
+</table>
+
+As you can see on this example, by adding 5-lines to any standard PyTorch training script you can now run on any kind of single or distributed node setting (single CPU, single GPU, multi-GPUs and TPUs) as well as with or without mixed precision (fp16).
+
+The same code can then in paticular run without modification on your local machine for debugging or your training environment.
 
 🤗 Accelerate also provides a CLI tool that allows you to quickly configure and test your training environment then launch the scripts.
 
diff --git a/examples/README.md b/examples/README.md
new file mode 100644
index 000000000..cb1d2645b
--- /dev/null
+++ b/examples/README.md
@@ -0,0 +1,75 @@
+# In this folder we showcase various full examples using `Accelerate`
+
+## Simple NLP example
+
+The [simple_example.py](./simple_example.py) script is a simple example to train a Bert model on a classification task ([GLUE's MRPC]()).
+
+The same script can be run in any of the following configurations:
+- single CPU or single GPU
+- multi GPUS (using PyTorch distributed mode)
+- (multi) TPUs
+- fp16 (mixed-precision) or fp32 (normal precision)
+
+To run it in each of these various modes, use the following commands:
+- single CPU:
+    * from a server without GPU
+        ```bash
+        python ./simple_example.py
+        ```
+    * from any server by passing `cpu=True` to the `Accelerator`.
+        ```bash
+        python ./simple_example.py --cpu
+        ```
+    * from any server with Accelerate launcher
+        ```bash
+        accelerate launch --cpu ./simple_example.py
+        ```
+- single GPU:
+    ```bash
+    python ./simple_example.py  # from a server with a GPU
+    ```
+- with fp16 (mixed-precision)
+    * from any server by passing `fp16=True` to the `Accelerator`.
+        ```bash
+        python ./simple_example.py --fp16
+        ```
+    * from any server with Accelerate launcher
+        ```bash
+        accelerate launch --fb16 ./simple_example.py
+- multi GPUS (using PyTorch distributed mode)
+    * With Accelerate config and launcher
+        ```bash
+        accelerate config  # This will create a config file on your server
+        accelerate launch ./simple_example.py  # This will run the script on your server
+        ```
+    * With traditional PyTorch launcher
+        ```bash
+        python -m torch.distributed.launch --nproc_per_node 2 --use_env ./simple_example.py
+        ```
+- multi GPUs, multi node (several machines, using PyTorch distributed mode)
+    * With Accelerate config and launcher, on each machine:
+        ```bash
+        accelerate config  # This will create a config file on each server
+        accelerate launch ./simple_example.py  # This will run the script on each server
+        ```
+    * With PyTorch launcher only
+        ```bash
+        python -m torch.distributed.launch --nproc_per_node 2 \
+            --use_env \
+            --node_rank 0 \
+            --master_addr master_node_ip_address \
+            ./simple_example.py  # On the first server
+        python -m torch.distributed.launch --nproc_per_node 2 \
+            --use_env \
+            --node_rank 1 \
+            --master_addr master_node_ip_address \
+            ./simple_example.py  # On the second server
+        ```
+- (multi) TPUs
+    * With Accelerate config and launcher
+        ```bash
+        accelerate config  # This will create a config file on your TPU server
+        accelerate launch ./simple_example.py  # This will run the script on each server
+        ```
+    * In PyTorch:
+        Add an `xmp.spawn` line in your script as you usually do.
diff --git a/examples/simple_example.py b/examples/simple_example.py
index dbbfd22f8..88c81e3a0 100644
--- a/examples/simple_example.py
+++ b/examples/simple_example.py
@@ -11,11 +11,28 @@
 )
```


# Output: a full review

The following is what the `POST` request should be made of in order to create an Review.
```json
{
    "body":
        'This is close to perfect! Please address the suggested inline change.',
    "comments" : [
        {
            path: 'file.md',
            position: 6,
            body: 'Please add more information here, and fix this typo.'
        }
    ]
}
```
The actual output of the model does not necessarily has to be this.

## The labels from github
```json
{
    "diff": 'diff --git a/src/accelerate/utils/modeling.py b/src/accelerate/utils/modeling.py\nindex c58cfeb50..679e57ff2 100644\n--- a/src/accelerate/utils/modeling.py\n+++ b/src/accelerate/utils/modeling.py\n@@ -666,7 +666,7 @@ def load_checkpoint_in_model(\n         elif len(potential_index) == 1:\n             index_filename = os.path.join(checkpoint, potential_index[0])\n         else:\n-            raise ValueError(f"{checkpoint} containing mote than one `.index.json` file, delete the irrelevant ones.")\n+            raise ValueError(f"{checkpoint} containing more than one `.index.json` file, delete the irrelevant ones.")\n     else:\n         raise ValueError(\n             "`checkpoint` should be the path to a file containing a whole state dict, or the index of a sharded "\n',
 "code_comments": [
    {"body": "Might be clearer to have an `else` block here instead of the early `return`.", "diff_hunk": "@@ -529,6 +530,23 @@ def set_deepspeed_weakref(self):\n \n             self.dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive # noqa\n \n+    def is_zero3_init_enabled(self):\n+        return self.zero3_init_flag\n+\n+    @contextmanager\n+    def set_zero3_init(self, enable=False):\n+        old = self.zero3_init_flag\n+        if old == enable:\n+            yield\n+            return\n+        self.zero3_init_flag = enable\n+        self.dschf = None\n+        self.set_deepspeed_weakref()\n+        yield\n+        self.zero3_init_flag = old\n+        self.dschf = None\n+        self.set_deepspeed_weakref()", "from_author": false}, {"body": "Done. ", "diff_hunk": "@@ -529,6 +530,23 @@ def set_deepspeed_weakref(self):\n \n             self.dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive # noqa\n \n+    def is_zero3_init_enabled(self):\n+        return self.zero3_init_flag\n+\n+    @contextmanager\n+    def set_zero3_init(self, enable=False):\n+        old = self.zero3_init_flag\n+        if old == enable:\n+            yield\n+            return\n+        self.zero3_init_flag = enable\n+        self.dschf = None\n+        self.set_deepspeed_weakref()\n+        yield\n+        self.zero3_init_flag = old\n+        self.dschf = None\n+        self.set_deepspeed_weakref()", "from_author": true}
 ],
 "context": [
    {'id': 1374301240,
   'created_at': datetime.datetime(2023, 1, 7, 0, 22, 51),
   'body': '_The documentation is not available anymore as the PR was closed or merged._',
   'from_author': False}
   ]
}
```
- The `context` are all of the `comments` made on the PR. We want to predict them.
- The `code_coments` are the suggestions
- The diff is the full PR's diff.
- We don't need the `id` is not important, while we need the `from_author` just for filtering.



# Format of the dataset

```json
[{
    "pr_id" : 180928,
    "label" : {
        "code_comments": [
            {
                "body": "Might be clearer to have an `else` block here instead of the early `return`.", "diff_hunk": "@@ -529,6 +530,23 @@ def set_deepspeed_weakref(self):\n \n             self.dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive # noqa\n \n+    def is_zero3_init_enabled(self):\n+        return self.zero3_init_flag\n+\n+    @contextmanager\n+    def set_zero3_init(self, enable=False):\n+        old = self.zero3_init_flag\n+        if old == enable:\n+            yield\n+            return\n+        self.zero3_init_flag = enable\n+        self.dschf = None\n+        self.set_deepspeed_weakref()\n+        yield\n+        self.zero3_init_flag = old\n+        self.dschf = None\n+        self.set_deepspeed_weakref()"
            }
        ],
        "context": [
            '_The documentation is not available anymore as the PR was closed or merged._',
            'This is interesting'
        ]
    },
    "raw_diff": """diff --git a/.gitignore b/.gitignore
index b6e47617d..3b89c957d 100644
--- a/.gitignore
+++ b/.gitignore
@@ -127,3 +127,6 @@ dmypy.json
 
 # Pyre type checker
 .pyre/
+
+# VSCode
+.vscode
diff --git a/README.md b/README.md
index 781eea3c9..fcc20a12a 100644
--- a/README.md
+++ b/README.md
@@ -34,7 +34,98 @@
 <p>Run your *raw* PyTorch training script on any kind of device
 </h3>
 
-🤗 Accelerate provides an easy API to make your scripts run with mixed precision and on any kind of distributed setting (multi-GPUs, TPUs etc.) while still letting you write your own training loop. The same code can then runs seamlessly on your local machine for debugging or your training environment.
+🤗 Accelerate was created for PyTorch users who like to write the training loop of PyTorch models but are reluctant to write and maintain the boiler code needed to use multi-GPUs/TPU/fp16.
+
+🤗 Accelerate abstracts exactly and only the boiler code related to multi-GPUs/TPU/fp16 and let the rest of your code unchanged.
+"""
},
]
