+++
author = "Laurens Weitkamp"
title = "Optimizing Transformers Series"
date = "2023-04-07"
description = ""
tags = []
+++

Brief notes along with [code](https://github.com/lweitkamp/optimizing_transformers) for common ways to squeeze performance out of transformer training and inference. 

**[Packing]({{< ref "/posts/packing.md" >}} "Packing").** Packing multiple sequences into a single sample reduces padding and increase the effective batch size during training.

