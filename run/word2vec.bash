#!/bin/bash
python -m src.word2vector.run $1
python -m src.word2vector.report $1
