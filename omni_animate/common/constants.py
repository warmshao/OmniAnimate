# -*- coding: utf-8 -*-
# @Time    : 2024/10/8 10:37
# @Author  : wenshao
# @ProjectName: ChatTTSPlus
# @FileName: constants.py

import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
CHECKPOINT_DIR = os.environ.get("OMNIANIMATE_CHECKPOINT_DIR",
                                os.path.abspath(os.path.join(PACKAGE_DIR, 'checkpoints')))
CHECKPOINT_DIR = os.path.abspath(CHECKPOINT_DIR)
