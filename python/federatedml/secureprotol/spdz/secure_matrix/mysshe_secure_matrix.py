#
#  Copyright 2021 The FATE Authors. All Rights Reserved.
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
#


import numpy as np

from fate_arch.common import Party
from fate_arch.session import is_table, computing_session
from federatedml.secureprotol.fixedpoint import FixedPointEndec
from federatedml.secureprotol.spdz.tensor import fixedpoint_numpy, fixedpoint_table
from federatedml.transfer_variable.transfer_class.secret_share_transfer_variable import SecretShareTransferVariable
from federatedml.util import consts, LOGGER

from federatedml.secureprotol.spdz.utils import rand_tensor, urand_tensor
from federatedml.secureprotol.fixedpoint import FixedPointNumber

import os
import zmq
import time
import zlib
import copy
import math
import pickle
import struct
import hashlib
import threading

HIGH_WATER_MARK = 15000
SUBSCRIBER_URL = 'tcp://192.168.122.12:6001'
PUBLISHER_URL = 'tcp://192.168.122.12:6000'

class SecureMatrix(object):
    # SecureMatrix in SecretSharing With He;
    def __init__(self, party: Party, q_field, other_party):
        self.transfer_variable = SecretShareTransferVariable()
        self.party = party
        self.other_party = other_party
        self.q_field = q_field
        self.encoder = None
        self.get_or_create_endec(self.q_field)

    def set_flowid(self, flowid):
        self.transfer_variable.set_flowid(flowid)

    def get_or_create_endec(self, q_field, **kwargs):
        if self.encoder is None:
            self.encoder = FixedPointEndec(q_field)
        return self.encoder

    #def secure_matrix_mul(self, matrix, tensor_name, cipher=None, suffix=tuple(), is_fixedpoint_table=True):
    #    current_suffix = ("secure_matrix_mul",) + suffix
    #    dst_role = consts.GUEST if self.party.role == consts.HOST else consts.HOST

    #    if cipher is not None:
    #        de_matrix = self.encoder.decode(matrix.value)
    #        if isinstance(matrix, fixedpoint_table.FixedPointTensor):
    #            encrypt_mat = cipher.distribute_encrypt(de_matrix)
    #        else:
    #            encrypt_mat = cipher.recursive_encrypt(de_matrix)

    #        # remote encrypted matrix;
    #        self.transfer_variable.encrypted_share_matrix.remote(encrypt_mat,
    #                                                             role=dst_role,
    #                                                             idx=0,
    #                                                             suffix=current_suffix)

    #        share_tensor = SecureMatrix.from_source(tensor_name,
    #                                                self.other_party,
    #                                                cipher,
    #                                                self.q_field,
    #                                                self.encoder,
    #                                                is_fixedpoint_table=is_fixedpoint_table)

    #        return share_tensor

    #    else:
    #        share = self.transfer_variable.encrypted_share_matrix.get(role=dst_role,
    #                                                                  idx=0,
    #                                                                  suffix=current_suffix)

    #        if is_table(share):
    #            share = fixedpoint_table.PaillierFixedPointTensor(share)

    #            ret = share.dot(matrix)
    #        else:
    #            share = fixedpoint_numpy.PaillierFixedPointTensor(share)
    #            ret = share.dot(matrix)

    #        share_tensor = SecureMatrix.from_source(tensor_name,
    #                                                ret,
    #                                                cipher,
    #                                                self.q_field,
    #                                                self.encoder)

    #        return share_tensor

    #def share_encrypted_matrix(self, suffix, is_remote, cipher, **kwargs):
    #    current_suffix = ("share_encrypted_matrix",) + suffix
    #    if is_remote:
    #        for var_name, var in kwargs.items():
    #            dst_role = consts.GUEST if self.party.role == consts.HOST else consts.HOST
    #            if isinstance(var, fixedpoint_table.FixedPointTensor):
    #                encrypt_var = cipher.distribute_encrypt(var.value)
    #            else:
    #                encrypt_var = cipher.recursive_encrypt(var.value)
    #            self.transfer_variable.encrypted_share_matrix.remote(encrypt_var, role=dst_role,
    #                                                                 suffix=(var_name,) + current_suffix)
    #    else:
    #        res = []
    #        for var_name in kwargs.keys():
    #            dst_role = consts.GUEST if self.party.role == consts.HOST else consts.HOST
    #            z = self.transfer_variable.encrypted_share_matrix.get(role=dst_role, idx=0,
    #                                                                  suffix=(var_name,) + current_suffix)
    #            if is_table(z):
    #                res.append(fixedpoint_table.PaillierFixedPointTensor(z))
    #            else:
    #                res.append(fixedpoint_numpy.PaillierFixedPointTensor(z))

    #        return tuple(res)

    def mysshe_matrix_mul(self, matrix1, matrix2, session_id, bts, suffix=tuple(), aggregate=False, ready=False):                                                                                                  
        current_suffix = ("mysshe_matrix_mul",) + suffix
        dst_role = consts.GUEST if self.party.role == consts.HOST else consts.HOST                                                                                                                                 

        if isinstance(matrix1, (fixedpoint_numpy.FixedPointTensor, fixedpoint_table.FixedPointTensor)):                                                                                                            
            matrix1_half = matrix1.value
        else:
            matrix1_half = matrix1                                                                                                                                                                                 

        if matrix2 is None:   
            matrix2_half = self.transfer_variable.share.get(role=dst_role,                 
                                                            idx=0,                         
                                                            suffix=current_suffix)                                                                                                                                 
        else:
            if isinstance(matrix2, (fixedpoint_numpy.FixedPointTensor, fixedpoint_table.FixedPointTensor)):                                                                                                        
                matrix2_value = matrix2.value
            else:
                matrix2_value = matrix2                                                                                                                                                                            

            if ready:
                matrix2_half = matrix2_value
            else:
                matrix2_share = self.rand_tensor(10, matrix2_value)                                                                                                                                                

                if is_table(matrix2_value):        
                    matrix2_half = matrix2_value.join(matrix2_share, lambda v1, v2: v1 - v2)                                                                                                                       
                else:
                    matrix2_half = matrix2_value - matrix2_share                                                                                                                                                   

                self.transfer_variable.share.remote(matrix2_share,                 
                                                    role=dst_role,                 
                                                    idx=0,                         
                                                    suffix=current_suffix)                                                                                                                                         
            
        LOGGER.info('secure matrix') 

        if is_table(matrix1_half):         
            if is_table(matrix2_half):         
                res = matrix1_half.join(matrix2_half, lambda v1, v2: (v1, v2))\
                                  .applyPartitions(lambda iterator: self.handle(iterator, None, bts, session_id, self.party.role, ''.join(current_suffix), aggregate))\                                            
                                  .flatMap(lambda k, v: v)                                                                                                                                                         
            else:
                res = matrix1_half.applyPartitions(lambda iterator: self.handle(iterator, matrix2_half, bts, session_id, self.party.role, ''.join(current_suffix), aggregate)).flatMap(lambda k, v: v)             
        else:
            res = matrix2_half.applyPartitions(lambda iterator: self.handle(iterator, matrix1_half, bts, session_id, self.party.role, ''.join(current_suffix), aggregate)).flatMap(lambda k, v: v)

        return fixedpoint_table.FixedPointTensor(res, self.q_field, self.encoder, tensor_name=''.join(current_suffix))

    @classmethod
    def from_source(cls, tensor_name, source, cipher, q_field, encoder, is_fixedpoint_table=True):
        if is_table(source):
            share_tensor = fixedpoint_table.PaillierFixedPointTensor.from_source(tensor_name=tensor_name,
                                                                                 source=source,
                                                                                 encoder=encoder,
                                                                                 q_field=q_field)
            return share_tensor

        elif isinstance(source, np.ndarray):
            share_tensor = fixedpoint_numpy.PaillierFixedPointTensor.from_source(tensor_name=tensor_name,
                                                                                 source=source,
                                                                                 encoder=encoder,
                                                                                 q_field=q_field)
            return share_tensor

        elif isinstance(source, (fixedpoint_table.PaillierFixedPointTensor,
                                 fixedpoint_numpy.PaillierFixedPointTensor)):
            return cls.from_source(tensor_name, source.value, cipher, q_field, encoder, is_fixedpoint_table)

        elif isinstance(source, Party):
            if is_fixedpoint_table:
                share_tensor = fixedpoint_table.PaillierFixedPointTensor.from_source(tensor_name=tensor_name,
                                                                                     source=source,
                                                                                     encoder=encoder,
                                                                                     q_field=q_field,
                                                                                     cipher=cipher)
            else:
                share_tensor = fixedpoint_numpy.PaillierFixedPointTensor.from_source(tensor_name=tensor_name,
                                                                                     source=source,
                                                                                     encoder=encoder,
                                                                                     q_field=q_field,
                                                                                     cipher=cipher)

            return share_tensor
        else:
            raise ValueError(f"type={type(source)}")

    def _generate_subscription(self, msg):
        return hashlib.md5(msg if isinstance(msg, bytes) else msg.encode('utf-8')).hexdigest().encode('utf-8')

    def handle(self, iterator, vector, bts, session_id, role, aggregate):
        def _generate_bt_tensor(bt):
            len_bt = len(bt)
            arr = np.zeros(shape=len_bt, dtype=FixedPointNumber)
            view = arr.view().reshape(-1)
            for i in range(len_bt):
                view[i] = FixedPointNumber.encode(bt[i])
            return arr

        def _send_data():
            for value in data.values():
                publisher.send(value['s_obj'])
            publisher.close()

        def _recv_data():
            count = 0
            sent = False
            send_thread = None
            len_data = len(data)

            while count < len_data or not sent:
                event = subscriber.poll(timeout=1000)
                if not event:
                    if not sent:
                        publisher.send(publisher_id)
                    continue

                msg = subscriber.recv()
                if 3 == msg[0]:
                    if not sent:
                        sent = True
                        subscriber.unsubscribe(msg)
                        send_thread = threading.Thread(target=_send_data)
                        send_thread.start()
                    continue

                topic, val = msg.split(b'_', 1)
                k = dict_subscription.pop(topic, None)
                if k is None:
                    continue
                data[k]['remote'] = val
                count += 1

            subscriber.close()

            if send_thread is not None:
                send_thread.join()

            ctx.destroy()

        subscription = self._generate_subscription(f'{session_id}-handle-{os.getpid()}')
        publisher_id = b''.join((b'\x03', subscription))

        ctx = zmq.Context().instance()

        subscriber = ctx.socket(zmq.SUB)
        subscriber.setsockopt(zmq.SNDHWM, HIGH_WATER_MARK)
        subscriber.setsockopt(zmq.RCVHWM, HIGH_WATER_MARK)
        subscriber.connect(SUBSCRIBER_URL)
        subscriber.subscribe(publisher_id)

        publisher = ctx.socket(zmq.PUB)
        publisher.setsockopt(zmq.SNDHWM, HIGH_WATER_MARK)
        publisher.setsockopt(zmq.RCVHWM, HIGH_WATER_MARK)
        publisher.connect(PUBLISHER_URL)

        remote_session_id = '_'.join((session_id.rsplit('_', 2)[0], self.other_party.role, self.other_party.party_id))

        dict_subscription = dict()
        data = dict()

        a_func = (lambda s, t: s[1][0] + t) if vector is None else (lambda s, t: s[1] + t)
        b_func = (lambda s, t: s[1][1] + t) if vector is None else (lambda s, t: vector + t)

        #for i, item in enumerate(iterator):
        for item in iterator:
            subscription = self._generate_subscription(f'{session_id}-handle-{item[0]}')
            dict_subscription[subscription] = item[0]
            subscriber.subscribe(subscription)

            a = _generate_bt_tensor(bts[item[0]]['a'])
            b = _generate_bt_tensor(bts[item[0]]['b'])
            if aggregate:
                c = np.array([FixedPointNumber.encode(sum(bts[item[0]]['c']))])
            else:
                c = _generate_bt_tensor(bts[item[0]]['c'])

            x = a_func(item, a)
            y = b_func(item, b)
            
            data_obj = zlib.compress(pickle.dumps(dict(x=x, y=y)))
            topic = self._generate_subscription(f'{remote_session_id}-handle-{item[0]}')
            data[item[0]] = dict(a=a, b=b, c=c, x=x, y=y, s_obj=b'_'.join((topic, data_obj)))


        _recv_data()

        result = list()
        for k, item in data.items():
            remote = pickle.loads(zlib.decompress(item['remote']))
            xx = item['x'] + remote['x']
            yy = item['y'] + remote['y']

            if aggregate:
                cross = item['c'] - np.dot(item['a'], yy) - np.dot(item['b'], xx)
                if consts.GUEST == role:
                    cross += np.dot(xx, yy)
            else:
                cross = item['c'] - item['a'] * yy - item['b'] * xx
                if consts.GUEST == role:
                    cross += xx * yy
            result.append((k, cross))

        return result
