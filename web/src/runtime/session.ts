import {Model,math,tf} from './tf';
import {createSession} from './session-core';
export {axis,type Baseline} from './session-core';
export const Session=createSession({Model,math,tensors:{
  tensor2d:tf.tensor2d,tensor1d:tf.tensor1d,tidy:tf.tidy,dispose:tf.dispose,
  add:tf.add,sub:tf.sub,mul:tf.mul,div:tf.div,exp:tf.exp,neg:tf.neg,square:tf.square,
  softplus:tf.softplus,sigmoid:tf.sigmoid,reshape:tf.reshape,slice:tf.slice,sum:tf.sum,
  clipByValue:tf.clipByValue,stack:tf.stack,gather:tf.gather,grad:tf.grad,
}});
export type Session=InstanceType<typeof Session>;
