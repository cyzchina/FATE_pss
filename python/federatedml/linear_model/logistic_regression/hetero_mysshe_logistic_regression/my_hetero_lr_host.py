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

import functools
import operator

import numpy as np

from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.linear_model.logistic_regression.hetero_mysshe_logistic_regression.my_hetero_lr_base import MyHeteroLRBase
from federatedml.protobuf.generated import lr_model_param_pb2
from federatedml.secureprotol.fate_paillier import PaillierPublicKey, PaillierEncryptedNumber
from federatedml.secureprotol.spdz.secure_matrix.secure_matrix import SecureMatrix
from federatedml.secureprotol.spdz.tensor import fixedpoint_table, fixedpoint_numpy
from federatedml.util import consts, LOGGER
from federatedml.util import fate_operator

from fate_arch.session import get_session

from federatedml.secureprotol.spdz import SPDZ

class MyHeteroLRHost(MyHeteroLRBase):
    def __init__(self):
        super().__init__()
        self.data_batch_count = []
        self.wx_self = None

    def _cal_z_in_sshe(self, w_self, w_remote, features, bts_self, bts_remote, suffix):
        session_id = get_session()._session_id

        '''
        t1 = w_self.value
        for item in t1:
            LOGGER.info(f'w_self: {item.decode()}')
        t1 = w_remote.value
        for item in t1:
            LOGGER.info(f'w_remote: {item.decode()}')
        '''

        za_suffix = ("za",) + suffix
        za_share = self.secure_matrix_obj.mysshe_matrix_mul(w_remote,
                                                            None,
                                                            session_id=session_id,
                                                            bts=bts_remote,
                                                            suffix=za_suffix,
                                                            aggregate=True)

        LOGGER.info('za done')

        zb_suffix = ("zb",) + suffix
        zb_share = self.secure_matrix_obj.mysshe_matrix_mul(w_self,
                                                            features,
                                                            session_id=session_id,
                                                            bts=bts_self,
                                                            suffix=zb_suffix,
                                                            aggregate=True)

        LOGGER.info('zb done')

        return za_share + zb_share

    def compute_shared_z(self, weights, features, bts_self, bts_remote, suffix):
        LOGGER.info(f"[forward]: Calculate z in share...")
        w_self, w_remote = weights
        return self._cal_z_in_sshe(w_self, w_remote, features, bts_self, bts_remote, suffix)

    def _cal_g_in_sshe(self, z, features, bts_self, bts_remote, suffix):
        session_id = get_session()._session_id

        ga_suffix = ("ga",) + suffix
        ga_share = self.secure_matrix_obj.mysshe_matrix_mul(z,
                                                            None,
                                                            session_id=session_id,
                                                            bts=bts_remote,
                                                            suffix=ga_suffix)\
                                         .reduce(lambda x, y: x + y)

        LOGGER.info('ga done')

        gb_suffix = ("gb",) + suffix
        gb_share = self.secure_matrix_obj.mysshe_matrix_mul(z,
                                                            features,
                                                            session_id=session_id,
                                                            bts=bts_self,
                                                            suffix=gb_suffix)\
                                         .reduce(lambda x, y: x + y)

        LOGGER.info('gb done')
        
        '''
        for item in ga_share.value:
            LOGGER.info(f'ga_share: {item.decode()}')
        for item in gb_share.value:
            LOGGER.info(f'gb_share: {item.decode()}')
        '''

        return gb_share, ga_share

    def compute_shared_g(self, shared_z, labels, features, bts_self, bts_remote, suffix):
        """ 
        y ϵ {-1, 1}, loss = (1 / n) * Σ ln(1 + exp(-yi * zi))
        loss' = (1/ n) * Σ ((1 / (1 + exp(-yi *zi))) - 1) * yi * xi
        ln(1 + exp(-z)) Taylor series expansion: ln2 - (1/2) * z + (1/8) * z² - (1/192) * z⁴ + O(z⁶)
        loss = (1 / n) * Σ (ln2 - (1/2) * yi * zi + (1/8) * (yi * zi)²) = (1 / n) * Σ (ln2 - (1/2) * yi * zi + (1/8) * zi²)
        loss' = (1 / n) * Σ ((1/4) * zi - (1/2) * yi) * xi
        """
        LOGGER.info(f"[forward]: Calculate g in share...")
        z = shared_z.value.mapValues(lambda v: v * 0.25)
        return self._cal_g_in_sshe(z, features, bts_self, bts_remote, suffix)

    def compute_shared_loss(self, shared_z, labels, bts_z_square, bts_z_label, suffix):
        """
          Use Taylor series expand log loss:
          Loss = - y * log(h(x)) - (1-y) * log(1 - h(x)) where h(x) = 1/(1+exp(-wx))
          Then loss' = - (1/N)*∑(log(1/2) - 1/2*wx + ywx -1/8(wx)^2)
        """
        LOGGER.info(f"[compute_loss]: Calculate loss ...")
        z_square_share = self._cal_z_square_in_sshe(shared_z, bts_z_square, suffix)
        z_label_share = self._cal_z_label_in_sshe(shared_z, labels, bts_z_label, suffix)
        loss = z_square_share * 0.125 + z_label_share
        self.secure_matrix_obj.transfer_variable.share.remote(loss,
                                                              role=consts.GUEST,
                                                              idx=0,
                                                              suffix=('loss',) + suffix)

    #def compute_loss(self, weights=None, labels=None, suffix=None, cipher=None):
    #    """
    #      Use Taylor series expand log loss:
    #      Loss = - y * log(h(x)) - (1-y) * log(1 - h(x)) where h(x) = 1/(1+exp(-wx))
    #      Then loss' = - (1/N)*∑(log(1/2) - 1/2*wx + ywx - 1/8(wx)^2)
    #    """
    #    LOGGER.info(f"[compute_loss]: Calculate loss ...")
    #    wx_self_square = (self.wx_self * self.wx_self).reduce(operator.add)

    #    self.secure_matrix_obj.share_encrypted_matrix(suffix=suffix,
    #                                                  is_remote=True,
    #                                                  cipher=cipher,
    #                                                  wx_self_square=wx_self_square)

    #    tensor_name = ".".join(("shared_loss",) + suffix)
    #    share_loss = SecureMatrix.from_source(tensor_name=tensor_name,
    #                                          source=self.other_party,
    #                                          cipher=cipher,
    #                                          q_field=self.fixedpoint_encoder.n,
    #                                          encoder=self.fixedpoint_encoder,
    #                                          is_fixedpoint_table=False)

    #    if self.reveal_every_iter:
    #        loss_norm = self.optimizer.loss_norm(weights)
    #        if loss_norm:
    #            share_loss += loss_norm
    #        LOGGER.debug(f"share_loss+loss_norm: {share_loss}")
    #        tensor_name = ".".join(("loss",) + suffix)
    #        share_loss.broadcast_reconstruct_share(tensor_name=tensor_name)
    #    else:
    #        tensor_name = ".".join(("loss",) + suffix)
    #        share_loss.broadcast_reconstruct_share(tensor_name=tensor_name)
    #        if self.optimizer.penalty == consts.L2_PENALTY:
    #            w_self, w_remote = weights

    #            w_encode = np.hstack((w_self.value, w_remote.value))

    #            w_encode = np.array([w_encode])

    #            w_tensor_name = ".".join(("loss_norm_w",) + suffix)
    #            w_tensor = fixedpoint_numpy.FixedPointTensor(value=w_encode,
    #                                                         q_field=self.fixedpoint_encoder.n,
    #                                                         endec=self.fixedpoint_encoder,
    #                                                         tensor_name=w_tensor_name)

    #            w_tensor_transpose_name = ".".join(("loss_norm_w_transpose",) + suffix)
    #            w_tensor_transpose = fixedpoint_numpy.FixedPointTensor(value=w_encode.T,
    #                                                                   q_field=self.fixedpoint_encoder.n,
    #                                                                   endec=self.fixedpoint_encoder,
    #                                                                   tensor_name=w_tensor_transpose_name)

    #            loss_norm_tensor_name = ".".join(("loss_norm",) + suffix)

    #            loss_norm = w_tensor.dot(w_tensor_transpose, target_name=loss_norm_tensor_name)
    #            loss_norm.broadcast_reconstruct_share()

    def _reveal_every_iter_weights_check(self, last_w, new_w, suffix):
        square_sum = np.sum((last_w - new_w) ** 2)
        self.converge_transfer_variable.square_sum.remote(square_sum, role=consts.GUEST, idx=0, suffix=suffix)
        return self.converge_transfer_variable.converge_info.get(idx=0, suffix=suffix)

    def predict(self, data_instances):
        return self.predict_tradition(data_instances)

    def predict_tradition(self, data_instances):
        LOGGER.info("Start predict ...")
        self._abnormal_detection(data_instances)
        data_instances = self.align_data_header(data_instances, self.header)
        if self.need_one_vs_rest:
            self.one_vs_rest_obj.predict(data_instances)
            return

        LOGGER.debug(f"Before_predict_reveal_strategy: {self.model_param.reveal_strategy},"
                     f" {self.is_respectively_reveal}")

        def _vec_dot(v, coef, intercept):
            return fate_operator.vec_dot(v.features, coef) + intercept

        f = functools.partial(_vec_dot,
                              coef=self.model_weights.coef_,
                              intercept=self.model_weights.intercept_)
        prob_host = data_instances.mapValues(f)
        self.transfer_variable.host_prob.remote(prob_host, role=consts.GUEST, idx=0)
        LOGGER.info("Remote probability to Guest")

    def get_single_model_param(self, model_weights=None, header=None):
        result = super().get_single_model_param(model_weights, header)
        if not self.is_respectively_reveal:
            weight_dict = {}
            model_weights = model_weights if model_weights else self.model_weights
            header = header if header else self.header
            for idx, header_name in enumerate(header):
                coef_i = model_weights.coef_[idx]

                is_obfuscator = False
                if hasattr(coef_i, "__is_obfuscator"):
                    is_obfuscator = getattr(coef_i, "__is_obfuscator")

                public_key = lr_model_param_pb2.CipherPublicKey(n=str(coef_i.public_key.n))
                weight_dict[header_name] = lr_model_param_pb2.CipherText(public_key=public_key,
                                                                         cipher_text=str(coef_i.ciphertext()),
                                                                         exponent=str(coef_i.exponent),
                                                                         is_obfuscator=is_obfuscator)
            result["encrypted_weight"] = weight_dict

        return result

    def _get_param(self):
        if self.need_cv:
            param_protobuf_obj = lr_model_param_pb2.LRModelParam()
            return param_protobuf_obj

        self.header = self.header if self.header else []
        LOGGER.debug("In get_param, self.need_one_vs_rest: {}".format(self.need_one_vs_rest))

        if self.need_one_vs_rest:
            one_vs_rest_result = self.one_vs_rest_obj.save(lr_model_param_pb2.SingleModel)
            single_result = {'header': self.header, 'need_one_vs_rest': True, "best_iteration": -1}
        else:
            one_vs_rest_result = None
            single_result = self.get_single_model_param()
            single_result['need_one_vs_rest'] = False
        single_result['one_vs_rest_result'] = one_vs_rest_result

        param_protobuf_obj = lr_model_param_pb2.LRModelParam(**single_result)

        return param_protobuf_obj

    def load_single_model(self, single_model_obj):
        super(MyHeteroLRHost, self).load_single_model(single_model_obj)
        if not self.is_respectively_reveal:
            feature_shape = len(self.header)
            tmp_vars = [None] * feature_shape
            weight_dict = dict(single_model_obj.encrypted_weight)
            for idx, header_name in enumerate(self.header):
                cipher_weight = weight_dict.get(header_name)
                public_key = PaillierPublicKey(int(cipher_weight.public_key.n))
                cipher_text = int(cipher_weight.cipher_text)
                exponent = int(cipher_weight.exponent)
                is_obfuscator = cipher_weight.is_obfuscator
                coef_i = PaillierEncryptedNumber(public_key, cipher_text, exponent)
                if is_obfuscator:
                    coef_i.apply_obfuscator()

                tmp_vars[idx] = coef_i

            self.model_weights = LinearModelWeights(tmp_vars, fit_intercept=self.fit_intercept)

