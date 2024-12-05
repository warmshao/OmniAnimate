# -*- coding: utf-8 -*-
# @Time    : 2024/10/8 10:37
# @Author  : wenshao
# @ProjectName: ChatTTSPlus
# @FileName: constants.py

import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.environ.get("OMNIANIMATE_PROJECT_DIR", os.path.join(CURRENT_DIR, '..', '..'))
PROJECT_DIR = os.path.abspath(PROJECT_DIR)
CHECKPOINT_DIR = os.environ.get("OMNIANIMATE_CHECKPOINT_DIR",
                                os.path.abspath(os.path.join(CURRENT_DIR, '..', 'checkpoints')))
CHECKPOINT_DIR = os.path.abspath(CHECKPOINT_DIR)
LOG_DIR = os.environ.get("OMNIANIMATE_LOG_DIR", os.path.join(PROJECT_DIR, 'logs'))
LOG_DIR = os.path.abspath(LOG_DIR)
