import {Model,math,tensors} from './jax';
import {createSession} from './session-core';
export {axis,type Baseline} from './session-core';
export const Session=createSession({Model,math,tensors});
export type Session=InstanceType<typeof Session>;
