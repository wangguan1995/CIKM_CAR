# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code is heavily based on paper "Geometry-Informed Neural Operator for Large-Scale 3D PDEs", we use paddle to reproduce the results of the paper


from .cfd_datamodule import CFDSDFDataModule, CarDataModule


def instantiate_datamodule(config):
    if config.data_module == "CFDSDFDataModule":
        assert config.sdf_spatial_resolution is not None
        return CFDSDFDataModule(
            config.data_dir,
            config.n_train,
            config.n_test,
            spatial_resolution=config.sdf_spatial_resolution,
            test_data_dir=config.test_data_dir,
        )

    if config.data_module == "CarDataModule":
        assert config.sdf_spatial_resolution is not None
        return CarDataModule(
            config.data_dir,
            config.test_data_dir,
            config.n_train,
            config.n_test,
            spatial_resolution=config.sdf_spatial_resolution,
        )

    else:
        raise NotImplementedError(f"Unknown datamodule: {config.data_module}")
