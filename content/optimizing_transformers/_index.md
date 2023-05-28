+++
author = "Laurens Weitkamp"
title = "Optimizing Transformers Series"
date = "2023-04-07"
description = ""
tags = []
+++

Brief notes along with [code](https://github.com/lweitkamp/optimizing_transformers) for common ways to squeeze performance out of transformer training and inference. 

**[Packing]({{< ref "/posts/packing.md" >}} "Packing").** Packing multiple sequences into a single sample reduces padding and increase the effective batch size during training.

**[Key Value Cache and Multi Query Attention]({{< ref "/posts/kv_cache_and_multi_query_attention.md" >}} "KV Cache and Multi Query Attention").** Save compute during inference when autoregressively generating tokens by caching the key and value matrices. Comes with some caveats, the speed increase is only noticeable at a certain batch size dependent on the number of parameters. Multi Query attention is a way to drastically reduce memory requirements of KV caching, but we are reducing the parameter count of our model by doing so.
