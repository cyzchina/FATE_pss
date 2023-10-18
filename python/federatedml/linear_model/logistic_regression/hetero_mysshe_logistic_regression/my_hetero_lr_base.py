#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import copy
from abc import ABC

import math
import zlib
import time
import pickle
import random

import numpy as np

from redis import StrictRedis

from fate_arch.session import get_parties
from federatedml.framework.hetero.procedure import batch_generator
from federatedml.linear_model.linear_model_base import BaseLinearModel
from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.one_vs_rest.one_vs_rest import one_vs_rest_factory
from federatedml.param.hetero_sshe_lr_param import LogisticRegressionParam
from federatedml.param.logistic_regression_param import InitParam
from federatedml.protobuf.generated import lr_model_meta_pb2
from federatedml.secureprotol import EncryptModeCalculator
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.fixedpoint import FixedPointEndec
from federatedml.secureprotol.spdz import SPDZ
from federatedml.secureprotol.spdz.secure_matrix.mysshe_secure_matrix import SecureMatrix
from federatedml.secureprotol.spdz.tensor import fixedpoint_numpy, fixedpoint_table
from federatedml.transfer_variable.transfer_class.batch_generator_transfer_variable import \
    BatchGeneratorTransferVariable
from federatedml.transfer_variable.transfer_class.converge_checker_transfer_variable import \
    ConvergeCheckerTransferVariable
from federatedml.transfer_variable.transfer_class.sshe_model_transfer_variable import SSHEModelTransferVariable
from federatedml.util import LOGGER
from federatedml.util import consts

from federatedml.secureprotol.fixedpoint import FixedPointNumber
from fate_arch.session import get_session

class MyHeteroLRBase(BaseLinearModel, ABC):
    def __init__(self):
        super().__init__()
        self.model_name = 'HeteroMySSHELogisticRegression'
        self.model_param_name = 'HeteroMySSHELogisticRegressionParam'
        self.model_meta_name = 'HeteroMySSHELogisticRegressionMeta'
        self.mode = consts.HETERO
        self.cipher = None
        self.q_field = None
        self.model_param = LogisticRegressionParam()
        self.labels = None
        self.batch_num = []
        self.one_vs_rest_obj = None
        self.secure_matrix_obj: SecureMatrix
        self._set_parties()
        self.cipher_tool = None

    def adjust(self, item, suffix):
        if consts.GUEST == self.role:
            self.secure_matrix_obj.transfer_variable.share.remote(item,
                                                                  role=consts.HOST,
                                                                  idx=0,
                                                                  suffix=('guest_adjust',) + suffix)

            remote_item = self.secure_matrix_obj.transfer_variable.share.get(role=consts.HOST,
                                                                             idx=0,
                                                                             suffix=('host_adjust',) + suffix)
        else:
            remote_item = self.secure_matrix_obj.transfer_variable.share.get(role=consts.GUEST,
                                                                             idx=0,
                                                                             suffix=('guest_adjust',) + suffix)

            self.secure_matrix_obj.transfer_variable.share.remote(item,
                                                                  role=consts.GUEST,
                                                                  idx=0,
                                                                  suffix=('host_adjust',) + suffix)
        return remote_item

    def get_beaver_tripple(self, count, beaver_tripple_dict, suffix):
        def _scan_redis_keys():
            keys = []
            cursor = random.randint(1, 1000)
            while not keys:
                if 0 == cursor:
                    cursor = random.randint(1, 1000)
                    time.sleep(1)
                cursor, keys = r_conn.scan(cursor, match=f'{bt_pre}_*') 
            redis_keys.update(set(keys))

        def _get_data():
            val = None
            key = None
            while val is None:
                if not redis_keys:
                    _scan_redis_keys()

                key = redis_keys.pop()

                pipe.multi()
                pipe.get(key)
                pipe.delete(key)
                res = pipe.execute()

                val = res[0]
                if val is None:
                    redis_keys.clear()
                    _scan_redis_keys()
                else:
                    val = pickle.loads(zlib.decompress(val))
            return key, val

        bts = dict()
        keys = list()
        redis_keys = set()
        bts_current = dict(a=tuple(), b=tuple(), c=tuple())

        len_bts_count = 0
        bt_pre = f'bt{self.local_party.party_id}'

        with StrictRedis(host='redis') as r_conn:
            pipe = r_conn.pipeline(transaction=True)
            for k in beaver_tripple_dict:
                while len_bts_count < count:
                    redis_key, val = _get_data()
                    keys.append(redis_key)
                    len_bts_count += 4096
                    bts_current['a'] += val['a']
                    bts_current['b'] += val['b']
                    bts_current['c'] += val['c']
                bts[k] = dict(a=bts_current['a'][:count], b=bts_current['b'][:count], c=bts_current['c'][:count])
                bts_current['a'] = bts_current['a'][count:] 
                bts_current['b'] = bts_current['b'][count:]
                bts_current['c'] = bts_current['c'][count:]
                len_bts_count -= count

        dst_role = consts.GUEST if consts.HOST == self.role else consts.HOST
        self.secure_matrix_obj.transfer_variable.share.remote(keys,
                                                              role=dst_role,
                                                              idx=0,
                                                              suffix=("bt",) + suffix)
        return bts

    def load_beaver_tripple(self, count, beaver_tripple_dict, suffix):
        def _get_data(key):
            pipe.multi()
            pipe.get(key)
            pipe.delete(key)
            res = pipe.execute()
            val = pickle.loads(zlib.decompress(res[0]))
            return val

        dst_role = consts.GUEST if consts.HOST == self.role else consts.HOST
        keys = self.secure_matrix_obj.transfer_variable.share.get(role=dst_role,
                                                                  idx=0,
                                                                  suffix=("bt",) + suffix)

        bts = dict()
        bts_current = dict(a=tuple(), b=tuple(), c=tuple())
        len_bts_count = 0
        idx = 0
        
        with StrictRedis(host='redis') as r_conn:
            pipe = r_conn.pipeline(transaction=True)
            for k in beaver_tripple_dict:
                while len_bts_count < count:
                    val = _get_data(keys[idx])
                    idx += 1
                    len_bts_count += 4096
                    bts_current['a'] += val['a']
                    bts_current['b'] += val['b']
                    bts_current['c'] += val['c']
                bts[k] = dict(a=bts_current['a'][:count], b=bts_current['b'][:count], c=bts_current['c'][:count])
                bts_current['a'] = bts_current['a'][count:] 
                bts_current['b'] = bts_current['b'][count:]
                bts_current['c'] = bts_current['c'][count:]
                len_bts_count -= count
        return bts

    def _transfer_q_field(self):
        if self.role == consts.GUEST:
            q_field = self.cipher.public_key.n
            self.transfer_variable.q_field.remote(q_field, role=consts.HOST, suffix=("q_field",))

        else:
            q_field = self.transfer_variable.q_field.get(role=consts.GUEST, idx=0,
                                                          suffix=("q_field",))

        return q_field

    def _init_model(self, params: LogisticRegressionParam):
        super()._init_model(params)
        self.encrypted_mode_calculator_param = params.encrypted_mode_calculator_param
        if self.role == consts.HOST:
            self.init_param_obj.fit_intercept = False
        self.cipher = PaillierEncrypt()
        self.cipher.generate_key(self.model_param.encrypt_param.key_length)
        self.transfer_variable = SSHEModelTransferVariable()
        self.one_vs_rest_obj = one_vs_rest_factory(self, role=self.role, mode=self.mode, has_arbiter=False)

        self.converge_func_name = params.early_stop
        self.reveal_every_iter = params.reveal_every_iter

        self.q_field = self._transfer_q_field()

        LOGGER.debug(f"q_field: {self.q_field}")

        if not self.reveal_every_iter:
            self.self_optimizer = copy.deepcopy(self.optimizer)
            self.remote_optimizer = copy.deepcopy(self.optimizer)

        self.batch_generator = batch_generator.Guest() if self.role == consts.GUEST else batch_generator.Host()
        self.batch_generator.register_batch_generator(BatchGeneratorTransferVariable(), has_arbiter=False)
        self.fixedpoint_encoder = FixedPointEndec(n=self.q_field)
        self.converge_transfer_variable = ConvergeCheckerTransferVariable()
        self.secure_matrix_obj = SecureMatrix(party=self.local_party,
                                              q_field=self.q_field,
                                              other_party=self.other_party)

    def _init_weights(self, model_shape):
        return self.initializer.init_model(model_shape, init_params=self.init_param_obj)

    def _set_parties(self):
        parties = []
        guest_parties = get_parties().roles_to_parties(["guest"])
        host_parties = get_parties().roles_to_parties(["host"])
        parties.extend(guest_parties)
        parties.extend(host_parties)

        local_party = get_parties().local_party
        other_party = parties[0] if parties[0] != local_party else parties[1]

        self.parties = parties
        self.local_party = local_party
        self.other_party = other_party

    @property
    def is_respectively_reveal(self):
        return self.model_param.reveal_strategy == "respectively"

    def share_model(self, w, suffix):
        source = [w, self.other_party]
        if self.local_party.role == consts.GUEST:
            wb, wa = (
                fixedpoint_numpy.FixedPointTensor.from_source(f"wb_{suffix}", source[0],
                                                              encoder=self.fixedpoint_encoder,
                                                              q_field=self.q_field),
                fixedpoint_numpy.FixedPointTensor.from_source(f"wa_{suffix}", source[1],
                                                              encoder=self.fixedpoint_encoder,
                                                              q_field=self.q_field),
            )
            return wb, wa
        else:
            wa, wb = (
                fixedpoint_numpy.FixedPointTensor.from_source(f"wa_{suffix}", source[0],
                                                              encoder=self.fixedpoint_encoder,
                                                              q_field=self.q_field),
                fixedpoint_numpy.FixedPointTensor.from_source(f"wb_{suffix}", source[1],
                                                              encoder=self.fixedpoint_encoder,
                                                              q_field=self.q_field),
            )
            return wa, wb

    def _cal_z_square_in_sshe(self, shared_z, bts, suffix):
        session_id = get_session()._session_id

        '''
        for item in shared_z.value.collect():
            LOGGER.info(f'shared_z: {item[0]}, {item[1][0].decode()}') 
        '''

        z_square_suffix = ("z_square",) + suffix
        z_square_share = self.secure_matrix_obj.mysshe_matrix_mul(shared_z,
                                                                  shared_z,
                                                                  session_id=session_id,
                                                                  bts=bts,
                                                                  suffix=z_square_suffix,
                                                                  ready=True)\
                                               .reduce(lambda x, y: x + y)

        '''
        for item in z_square_share.value:
            LOGGER.info(f'z_square: {item.decode()}') 
        '''

        return z_square_share

    def _cal_z_label_in_sshe(self, shared_z, labels, bts, suffix):
        session_id = get_session()._session_id
        
        z_label_suffix = ("z_label",) + suffix
        z_label_share = self.secure_matrix_obj.mysshe_matrix_mul(shared_z,
                                                                 labels,
                                                                 session_id=session_id,
                                                                 bts=bts,
                                                                 suffix=z_label_suffix)\
                                               .reduce(lambda x, y: x + y)

        '''
        for item in z_label_share.value:
            LOGGER.info(f'z_label: {item.decode()}') 
        '''

        return z_label_share


    def compute_shared_z(self, weights, features, suffix):
        raise NotImplementedError("Should not call here")

    def compute_shared_g(self, shared_z, labels, features, suffix):
        raise NotImplementedError("Should not call here")

    def compute_loss(self, weights, labels, suffix, cipher):
        raise NotImplementedError("Should not call here")

    def compute_shared_loss(self, shared_z, labels, bts_z_square, bts_z_label, suffix):
        raise NotImplementedError("Should not call here")

    def fit(self, data_instances, validate_data=None):
        self.header = data_instances.schema.get("header", [])
        self._abnormal_detection(data_instances)
        self.check_abnormal_values(data_instances)
        self.check_abnormal_values(validate_data)
        classes = self.one_vs_rest_obj.get_data_classes(data_instances)

        if len(classes) > 2:
            self.need_one_vs_rest = True
            self.need_call_back_loss = False
            self.one_vs_rest_fit(train_data=data_instances, validate_data=validate_data)
        else:
            self.need_one_vs_rest = False
            self.fit_binary(data_instances, validate_data)

    def one_vs_rest_fit(self, train_data=None, validate_data=None):
        LOGGER.info("Class num larger than 2, do one_vs_rest")
        self.one_vs_rest_obj.fit(data_instances=train_data, validate_data=validate_data)

    def fit_binary(self, data_instances, validate_data=None):
        LOGGER.info("Starting to hetero_sshe_logistic_regression")
        self.callback_list.on_train_begin(data_instances, validate_data)

        model_shape = self.get_features_shape(data_instances)
        instances_count = data_instances.count()

        w = self._init_weights(model_shape)
        #if not self.component_properties.is_warm_start:
        #    w = self._init_weights(model_shape)
        #    self.model_weights = LinearModelWeights(l=w,
        #                                            fit_intercept=self.model_param.init_param.fit_intercept)
        #    last_models = copy.deepcopy(self.model_weights)
        #else:
        #    last_models = copy.deepcopy(self.model_weights)
        #    w = last_models.unboxed
        #    self.callback_warm_start_init_iter(self.n_iter_)

        self.batch_generator.initialize_batch_generator(data_instances, batch_size=self.batch_size)

        with SPDZ(
                "mysshe_lr",
                local_party=self.local_party,
                all_parties=self.parties,
                q_field=self.q_field,
                use_mix_rand=self.model_param.use_mix_rand,
        ) as spdz:
            spdz.set_flowid(self.flowid)
            self.secure_matrix_obj.set_flowid(self.flowid)
            w_self, w_remote = self.share_model(w, suffix="init")
            last_w_self, last_w_remote = w_self, w_remote
            LOGGER.debug(f"first_w_self shape: {w_self.shape}, w_remote_shape: {w_remote.shape}")

            batch_data_generator = self.batch_generator.generate_batch_data()

            self.cipher_tool = []
            encoded_batch_data = []
            batch_labels_list = []

            for batch_data in batch_data_generator:
                if self.fit_intercept:
                    batch_features = batch_data.mapValues(lambda x: np.hstack((x.features, 1.0)))
                else:
                    batch_features = batch_data.mapValues(lambda x: x.features)

                if self.role == consts.GUEST:
                    batch_labels = batch_data.mapValues(lambda x: np.array([x.label], dtype=int))
                    batch_labels_list.append(batch_labels)

                self.batch_num.append(batch_data.count())

                encoded_batch_data.append(
                    fixedpoint_table.FixedPointTensor(self.fixedpoint_encoder.encode(batch_features),
                                                      q_field=self.fixedpoint_encoder.n,
                                                      endec=self.fixedpoint_encoder))
                self.cipher_tool.append(EncryptModeCalculator(self.cipher,
                                                              self.encrypted_mode_calculator_param.mode,
                                                              self.encrypted_mode_calculator_param.re_encrypted_rate))

            beaver_tripple_self_count = w_self.shape[0]
            beaver_tripple_remote_count = w_remote.shape[0]
            while self.n_iter_ < self.max_iter:
                self.callback_list.on_epoch_begin(self.n_iter_)
                LOGGER.info(f"start to n_iter: {self.n_iter_}")

                loss_list = []

                self.optimizer.set_iters(self.n_iter_)
                if not self.reveal_every_iter:
                    self.self_optimizer.set_iters(self.n_iter_)
                    self.remote_optimizer.set_iters(self.n_iter_)

                for batch_idx, batch_data in enumerate(encoded_batch_data):
                    LOGGER.info(f"start to n_iter: {self.n_iter_}, batch idx: {batch_idx}")
                    current_suffix = (str(self.n_iter_), str(batch_idx))


                    beaver_tripple_dict = dict(batch_data.value.map(lambda k, v: (k, None)).collect())

                    if self.role == consts.GUEST:
                        batch_labels = batch_labels_list[batch_idx]

                        bts_z_self = self.get_beaver_tripple(beaver_tripple_self_count, beaver_tripple_dict, ("z1",) + current_suffix)
                        bts_z_remote = self.get_beaver_tripple(beaver_tripple_remote_count, beaver_tripple_dict, ("z2",) + current_suffix)

                        bts_g_self = self.get_beaver_tripple(beaver_tripple_self_count, beaver_tripple_dict, ("g1",) + current_suffix)
                        bts_g_remote = self.get_beaver_tripple(beaver_tripple_remote_count, beaver_tripple_dict, ("g2",) + current_suffix)

                        bts_z_square = self.get_beaver_tripple(1, beaver_tripple_dict, ("z_square",) + current_suffix)
                        bts_z_label = self.get_beaver_tripple(1, beaver_tripple_dict, ("z_label",) + current_suffix)
                    else:
                        batch_labels = None

                        bts_z_remote = self.load_beaver_tripple(beaver_tripple_remote_count, beaver_tripple_dict, ("z1",) + current_suffix)
                        bts_z_self = self.load_beaver_tripple(beaver_tripple_self_count, beaver_tripple_dict, ("z2",) + current_suffix)

                        bts_g_remote = self.load_beaver_tripple(beaver_tripple_remote_count, beaver_tripple_dict, ("g1",) + current_suffix)
                        bts_g_self = self.load_beaver_tripple(beaver_tripple_self_count, beaver_tripple_dict, ("g2",) + current_suffix)

                        bts_z_square = self.load_beaver_tripple(1, beaver_tripple_dict, ("z_square",) + current_suffix)
                        bts_z_label = self.load_beaver_tripple(1, beaver_tripple_dict, ("z_label",) + current_suffix)

                    shared_z = self.compute_shared_z(weights=(w_self, w_remote),
                                                     features=batch_data,
                                                     bts_self=bts_z_self,
                                                     bts_remote=bts_z_remote,
                                                     suffix=current_suffix)

                    g_self, g_remote = self.compute_shared_g(shared_z=shared_z,
                                                             labels=batch_labels,
                                                             features=batch_data,
                                                             bts_self=bts_g_self,
                                                             bts_remote=bts_g_remote,
                                                             suffix=current_suffix)
                    # loss computing;
                    suffix = ("loss",) + current_suffix
                    batch_loss = self.compute_shared_loss(shared_z=shared_z,
                                                          labels=batch_labels,
                                                          bts_z_square=bts_z_square,
                                                          bts_z_label=bts_z_label,
                                                          suffix=suffix)
                    if batch_loss is not None:
                        batch_loss -= np.log(0.5) * self.batch_num[batch_idx]
                        loss_list.append(batch_loss.value[0].decode())

                    self_g = g_self * (1 / self.batch_num[batch_idx])
                    remote_g = g_remote * (1 / self.batch_num[batch_idx])

                    adjust_self_g = list()
                    for g in self_g.value:
                        adjust_self_g.append(g.decode() // 10 * 10)

                    adjust_remote_g = self.adjust(adjust_self_g, suffix=("g",) + current_suffix)
                    #LOGGER.info(f'adjust_self_g: {adjust_self_g}')
                    #for item in self_g.value:
                    #    LOGGER.info(f'{item.decode()}')
                    self_g -= adjust_self_g
                    #for item in self_g.value:
                    #    LOGGER.info(f'{item.decode()}')
                    remote_g += adjust_remote_g

                    if consts.L2_PENALTY == self.optimizer.penalty:
                        self_g += self.self_optimizer.alpha * w_self
                        remote_g += self.remote_optimizer.alpha * w_remote

                    self_g = self.self_optimizer.apply_gradients(self_g)
                    remote_g = self.remote_optimizer.apply_gradients(remote_g)

                    w_self -= self_g
                    w_remote -= remote_g


                    '''
                    LOGGER.info(f'** {self.self_optimizer.alpha} {self.batch_num[batch_idx]} ')

                    t1 = w_self.value
                    for item in t1:
                        LOGGER.info(f'w_self: {item.decode()}')
                    t1 = w_remote.value
                    for item in t1:
                        LOGGER.info(f'w_remote: {item.decode()}')
                    '''
                    
                    LOGGER.debug(f"w_self shape: {w_self.shape}, w_remote_shape: {w_remote.shape}")

                if self.role == consts.GUEST:
                    loss = np.sum(loss_list) / instances_count
                    self.loss_history.append(loss)
                    if self.need_call_back_loss:
                        self.callback_loss(self.n_iter_, loss)
                else:
                    loss = None

                #if self.converge_func_name in ["diff", "abs"]:
                #    self.is_converged = self.check_converge_by_loss(loss, suffix=(str(self.n_iter_),))
                #elif self.converge_func_name == "weight_diff":
                #    if self.reveal_every_iter:
                #        self.is_converged = self.check_converge_by_weights(
                #            last_w=last_models.unboxed,
                #            new_w=self.model_weights.unboxed,
                #            suffix=(str(self.n_iter_),))
                #        last_models = copy.deepcopy(self.model_weights)
                #    else:
                #        self.is_converged = self.check_converge_by_weights(
                #            last_w=(last_w_self, last_w_remote),
                #            new_w=(w_self, w_remote),
                #            suffix=(str(self.n_iter_),))
                #        last_w_self, last_w_remote = copy.deepcopy(w_self), copy.deepcopy(w_remote)
                #else:
                #    raise ValueError(f"Cannot recognize early_stop function: {self.converge_func_name}")

                LOGGER.info("iter: {},  is_converged: {}".format(self.n_iter_, self.is_converged))
                self.callback_list.on_epoch_end(self.n_iter_)
                self.n_iter_ += 1

                if self.stop_training:
                    break

                if self.is_converged:
                    break

            self.w_self = w_self
            self.w_remote = w_remote

            self.fit_binary_tradition_finally(w_self, w_remote)

            # Finally reconstruct
            #if not self.reveal_every_iter:
            #    new_w = self.reveal_models(w_self, w_remote, suffix=("final",))
            #    if new_w is not None:
            #        self.model_weights = LinearModelWeights(
            #            l=new_w,
            #            fit_intercept=self.model_param.init_param.fit_intercept)

        LOGGER.debug(f"loss_history: {self.loss_history}")
        #self.set_summary(self.get_model_summary())

    def fit_binary_tradition_finally(self, w_self, w_remote):
        new_w = self.reveal_models(w_self, w_remote, suffix=("final",))
        if new_w is not None:
            self.model_weights = LinearModelWeights(
                l=new_w,
                fit_intercept=self.model_param.init_param.fit_intercept)
        self.set_summary(self.get_model_summary())

    def reveal_models(self, w_self, w_remote, suffix=None):
        if suffix is None:
            suffix = self.n_iter_

        if self.model_param.reveal_strategy == "respectively":

            if self.role == consts.GUEST:
                new_w = w_self.get(tensor_name=f"wb_{suffix}",
                                   broadcast=False)
                w_remote.broadcast_reconstruct_share(tensor_name=f"wa_{suffix}")

            else:
                w_remote.broadcast_reconstruct_share(tensor_name=f"wb_{suffix}")
                new_w = w_self.get(tensor_name=f"wa_{suffix}",
                                   broadcast=False)

        elif self.model_param.reveal_strategy == "encrypted_reveal_in_host":

            if self.role == consts.GUEST:
                new_w = w_self.get(tensor_name=f"wb_{suffix}",
                                   broadcast=False)
                encrypted_w_remote = self.cipher.recursive_encrypt(self.fixedpoint_encoder.decode(w_remote.value))
                encrypted_w_remote_tensor = fixedpoint_numpy.PaillierFixedPointTensor(value=encrypted_w_remote)
                encrypted_w_remote_tensor.broadcast_reconstruct_share(tensor_name=f"wa_{suffix}")
            else:
                w_remote.broadcast_reconstruct_share(tensor_name=f"wb_{suffix}")

                new_w = w_self.reconstruct(tensor_name=f"wa_{suffix}", broadcast=False)

        else:
            raise NotImplementedError(f"reveal strategy: {self.model_param.reveal_strategy} has not been implemented.")
        return new_w

    def check_converge_by_loss(self, loss, suffix):
        if self.role == consts.GUEST:
            self.is_converged = self.converge_func.is_converge(loss)
            self.transfer_variable.is_converged.remote(self.is_converged, suffix=suffix)
        else:
            self.is_converged = self.transfer_variable.is_converged.get(idx=0, suffix=suffix)
        return self.is_converged

    def check_converge_by_weights(self, last_w, new_w, suffix):
        if self.reveal_every_iter:
            return self._reveal_every_iter_weights_check(last_w, new_w, suffix)
        else:
            return self._not_reveal_every_iter_weights_check(last_w, new_w, suffix)

    def _reveal_every_iter_weights_check(self, last_w, new_w, suffix):
        raise NotImplementedError()

    def _not_reveal_every_iter_weights_check(self, last_w, new_w, suffix):
        last_w_self, last_w_remote = last_w
        w_self, w_remote = new_w
        grad_self = w_self - last_w_self
        grad_remote = w_remote - last_w_remote

        if self.role == consts.GUEST:
            grad_encode = np.hstack((grad_remote.value, grad_self.value))
        else:
            grad_encode = np.hstack((grad_self.value, grad_remote.value))

        grad_encode = np.array([grad_encode])

        grad_tensor_name = ".".join(("check_converge_grad",) + suffix)
        grad_tensor = fixedpoint_numpy.FixedPointTensor(value=grad_encode,
                                                        q_field=self.fixedpoint_encoder.n,
                                                        endec=self.fixedpoint_encoder,
                                                        tensor_name=grad_tensor_name)

        grad_tensor_transpose_name = ".".join(("check_converge_grad_transpose",) + suffix)
        grad_tensor_transpose = fixedpoint_numpy.FixedPointTensor(value=grad_encode.T,
                                                                  q_field=self.fixedpoint_encoder.n,
                                                                  endec=self.fixedpoint_encoder,
                                                                  tensor_name=grad_tensor_transpose_name)

        grad_norm_tensor_name = ".".join(("check_converge_grad_norm",) + suffix)

        grad_norm = grad_tensor.dot(grad_tensor_transpose, target_name=grad_norm_tensor_name).get()

        weight_diff = np.sqrt(grad_norm[0][0])
        LOGGER.info("iter: {}, weight_diff:{}, is_converged: {}".format(self.n_iter_,
                                                                        weight_diff, self.is_converged))
        is_converge = False
        if weight_diff < self.model_param.tol:
            is_converge = True
        return is_converge

    def _get_meta(self):
        meta_protobuf_obj = lr_model_meta_pb2.LRModelMeta(penalty=self.model_param.penalty,
                                                          tol=self.model_param.tol,
                                                          alpha=self.alpha,
                                                          optimizer=self.model_param.optimizer,
                                                          batch_size=self.batch_size,
                                                          learning_rate=self.model_param.learning_rate,
                                                          max_iter=self.max_iter,
                                                          early_stop=self.model_param.early_stop,
                                                          fit_intercept=self.fit_intercept,
                                                          need_one_vs_rest=self.need_one_vs_rest,
                                                          reveal_strategy=self.model_param.reveal_strategy)
        return meta_protobuf_obj

    def get_single_model_param(self, model_weights=None, header=None):
        header = header if header else self.header
        result = {'iters': self.n_iter_,
                  'loss_history': self.loss_history,
                  'is_converged': self.is_converged,
                  # 'weight': weight_dict,
                  #'intercept': self.model_weights.intercept_,
                  'intercept': self.w_self.value[-1].decode() if consts.GUEST == self.role and self.fit_intercept else 0.0,
                  'header': header,
                  'best_iteration': -1 if self.validation_strategy is None else
                  self.validation_strategy.best_iteration
                  }

        #if self.role == consts.GUEST or self.is_respectively_reveal:
        #    model_weights = model_weights if model_weights else self.model_weights
        #    weight_dict = {}
        #    for idx, header_name in enumerate(header):
        #        coef_i = model_weights.coef_[idx]
        #        weight_dict[header_name] = coef_i

        #    result['weight'] = weight_dict

        weight_dict = {}

        for idx, header_name in enumerate(header):
            weight_dict[f's_{header_name}'] = self.w_self.value[idx].decode()

        for idx, item in enumerate(self.w_remote.value):
            weight_dict[f'r_{idx}'] = item.decode()

        result['weight'] = weight_dict

        return result

    def get_model_summary(self):
        header = self.header
        if header is None:
            return {}
        weight_dict, intercept_ = self.get_weight_intercept_dict(header)
        best_iteration = -1 if self.validation_strategy is None else self.validation_strategy.best_iteration

        summary = {"coef": weight_dict,
                   "intercept": intercept_,
                   "is_converged": self.is_converged,
                   "one_vs_rest": self.need_one_vs_rest,
                   "best_iteration": best_iteration}

        if not self.is_respectively_reveal:
            del summary["intercept"]
            del summary["coef"]

        if self.validation_strategy:
            validation_summary = self.validation_strategy.summary()
            if validation_summary:
                summary["validation_metrics"] = validation_summary
        return summary

    def load_model(self, model_dict):
        LOGGER.debug("Start Loading model")
        result_obj = list(model_dict.get('model').values())[0].get(self.model_param_name)
        meta_obj = list(model_dict.get('model').values())[0].get(self.model_meta_name)

        if self.init_param_obj is None:
            self.init_param_obj = InitParam()
        self.init_param_obj.fit_intercept = meta_obj.fit_intercept
        self.model_param.reveal_strategy = meta_obj.reveal_strategy
        LOGGER.debug(f"reveal_strategy: {self.model_param.reveal_strategy}, {self.is_respectively_reveal}")
        self.header = list(result_obj.header)

        need_one_vs_rest = result_obj.need_one_vs_rest
        LOGGER.info("in _load_model need_one_vs_rest: {}".format(need_one_vs_rest))
        if need_one_vs_rest:
            one_vs_rest_result = result_obj.one_vs_rest_result
            self.one_vs_rest_obj = one_vs_rest_factory(classifier=self, role=self.role,
                                                       mode=self.mode, has_arbiter=False)
            self.one_vs_rest_obj.load_model(one_vs_rest_result)
            self.need_one_vs_rest = True
        else:
            self.load_single_model(result_obj)
            self.need_one_vs_rest = False

    def load_single_model(self, single_model_obj):
        LOGGER.info("It's a binary task, start to load single model")

        #if self.role == consts.GUEST or self.is_respectively_reveal:
        #    feature_shape = len(self.header)
        #    tmp_vars = np.zeros(feature_shape)
        #    weight_dict = dict(single_model_obj.weight)

        #    for idx, header_name in enumerate(self.header):
        #        tmp_vars[idx] = weight_dict.get(header_name)

        #    if self.fit_intercept:
        #        tmp_vars = np.append(tmp_vars, single_model_obj.intercept)
        #    self.model_weights = LinearModelWeights(tmp_vars, fit_intercept=self.fit_intercept)

        weight_dict = dict(single_model_obj.weight)

        w_self = list()
        for header_name in self.header:
            w_self.append(FixedPointNumber.encode(weight_dict.pop(f's_{header_name}', 0.0)))
        if consts.GUEST == self.role and self.fit_intercept:
            w_self.append(FixedPointNumber.encode(single_model_obj.intercept))
        self.w_self = fixedpoint_numpy.FixedPointTensor(value=np.array(w_self),
                                                        q_field=self.fixedpoint_encoder.n,
                                                        endec=self.fixedpoint_encoder,
                                                        tensor_name='w_self_init')

        len_remote = len(weight_dict)
        w_remote = [None] * len_remote
        for k, v in weight_dict.items():
            idx = int(k.split('_', 1)[-1])
            w_remote[idx] = FixedPointNumber.encode(v)
        self.w_remote = fixedpoint_numpy.FixedPointTensor(value=np.array(w_remote),
                                                          q_field=self.fixedpoint_encoder.n,
                                                          endec=self.fixedpoint_encoder,
                                                          tensor_name='w_remote_init')


        self.n_iter_ = single_model_obj.iters
        return self
