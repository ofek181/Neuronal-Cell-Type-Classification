��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02unknown8��
�
Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/v
y
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_23/kernel/v
�
*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v*
_output_shapes

:@*
dtype0
�
Adam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_22/bias/v
y
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes
:@*
dtype0
�
Adam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_22/kernel/v
�
*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v*
_output_shapes

:@@*
dtype0
�
"Adam/batch_normalization_17/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_17/beta/v
�
6Adam/batch_normalization_17/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_17/beta/v*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_17/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_17/gamma/v
�
7Adam/batch_normalization_17/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_17/gamma/v*
_output_shapes
:@*
dtype0
�
Adam/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_21/bias/v
y
(Adam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/v*
_output_shapes
:@*
dtype0
�
Adam/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_21/kernel/v
�
*Adam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/v*
_output_shapes

:@@*
dtype0
�
"Adam/batch_normalization_16/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_16/beta/v
�
6Adam/batch_normalization_16/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_16/beta/v*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_16/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_16/gamma/v
�
7Adam/batch_normalization_16/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_16/gamma/v*
_output_shapes
:@*
dtype0
�
Adam/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_20/bias/v
y
(Adam/dense_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/v*
_output_shapes
:@*
dtype0
�
Adam/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:)@*'
shared_nameAdam/dense_20/kernel/v
�
*Adam/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/v*
_output_shapes

:)@*
dtype0
�
"Adam/batch_normalization_15/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:)*3
shared_name$"Adam/batch_normalization_15/beta/v
�
6Adam/batch_normalization_15/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_15/beta/v*
_output_shapes
:)*
dtype0
�
#Adam/batch_normalization_15/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:)*4
shared_name%#Adam/batch_normalization_15/gamma/v
�
7Adam/batch_normalization_15/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_15/gamma/v*
_output_shapes
:)*
dtype0
�
Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/m
y
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_23/kernel/m
�
*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m*
_output_shapes

:@*
dtype0
�
Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_22/bias/m
y
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_22/kernel/m
�
*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m*
_output_shapes

:@@*
dtype0
�
"Adam/batch_normalization_17/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_17/beta/m
�
6Adam/batch_normalization_17/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_17/beta/m*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_17/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_17/gamma/m
�
7Adam/batch_normalization_17/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_17/gamma/m*
_output_shapes
:@*
dtype0
�
Adam/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_21/bias/m
y
(Adam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_21/kernel/m
�
*Adam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/m*
_output_shapes

:@@*
dtype0
�
"Adam/batch_normalization_16/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_16/beta/m
�
6Adam/batch_normalization_16/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_16/beta/m*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_16/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_16/gamma/m
�
7Adam/batch_normalization_16/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_16/gamma/m*
_output_shapes
:@*
dtype0
�
Adam/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_20/bias/m
y
(Adam/dense_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:)@*'
shared_nameAdam/dense_20/kernel/m
�
*Adam/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/m*
_output_shapes

:)@*
dtype0
�
"Adam/batch_normalization_15/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:)*3
shared_name$"Adam/batch_normalization_15/beta/m
�
6Adam/batch_normalization_15/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_15/beta/m*
_output_shapes
:)*
dtype0
�
#Adam/batch_normalization_15/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:)*4
shared_name%#Adam/batch_normalization_15/gamma/m
�
7Adam/batch_normalization_15/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_15/gamma/m*
_output_shapes
:)*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:*
dtype0
z
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_23/kernel
s
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes

:@*
dtype0
r
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_22/bias
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes
:@*
dtype0
z
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_22/kernel
s
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes

:@@*
dtype0
�
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_17/moving_variance
�
:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
_output_shapes
:@*
dtype0
�
"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_17/moving_mean
�
6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
_output_shapes
:@*
dtype0
�
batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_17/beta
�
/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_17/beta*
_output_shapes
:@*
dtype0
�
batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_17/gamma
�
0batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_17/gamma*
_output_shapes
:@*
dtype0
r
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
:@*
dtype0
z
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_21/kernel
s
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes

:@@*
dtype0
�
&batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_16/moving_variance
�
:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
_output_shapes
:@*
dtype0
�
"batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_16/moving_mean
�
6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes
:@*
dtype0
�
batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_16/beta
�
/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_16/beta*
_output_shapes
:@*
dtype0
�
batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_16/gamma
�
0batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_16/gamma*
_output_shapes
:@*
dtype0
r
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes
:@*
dtype0
z
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:)@* 
shared_namedense_20/kernel
s
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes

:)@*
dtype0
�
&batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:)*7
shared_name(&batch_normalization_15/moving_variance
�
:batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_15/moving_variance*
_output_shapes
:)*
dtype0
�
"batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:)*3
shared_name$"batch_normalization_15/moving_mean
�
6batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_15/moving_mean*
_output_shapes
:)*
dtype0
�
batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:)*,
shared_namebatch_normalization_15/beta
�
/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_15/beta*
_output_shapes
:)*
dtype0
�
batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:)*-
shared_namebatch_normalization_15/gamma
�
0batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_15/gamma*
_output_shapes
:)*
dtype0
�
,serving_default_batch_normalization_15_inputPlaceholder*'
_output_shapes
:���������)*
dtype0*
shape:���������)
�
StatefulPartitionedCallStatefulPartitionedCall,serving_default_batch_normalization_15_input"batch_normalization_15/moving_mean&batch_normalization_15/moving_variancebatch_normalization_15/betabatch_normalization_15/gammadense_20/kerneldense_20/bias"batch_normalization_16/moving_mean&batch_normalization_16/moving_variancebatch_normalization_16/betabatch_normalization_16/gammadense_21/kerneldense_21/bias"batch_normalization_17/moving_mean&batch_normalization_17/moving_variancebatch_normalization_17/betabatch_normalization_17/gammadense_22/kerneldense_22/biasdense_23/kerneldense_23/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_68789

NoOpNoOp
�u
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�u
value�uB�u B�u
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
axis
	gamma
beta
moving_mean
moving_variance*
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias*
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-_random_generator* 
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4axis
	5gamma
6beta
7moving_mean
8moving_variance*
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias*
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
G_random_generator* 
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance*
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias*
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
a_random_generator* 
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias*
�
0
1
2
3
%4
&5
56
67
78
89
?10
@11
O12
P13
Q14
R15
Y16
Z17
h18
i19*
j
0
1
%2
&3
54
65
?6
@7
O8
P9
Y10
Z11
h12
i13*
,
j0
k1
l2
m3
n4
o5* 
�
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
utrace_0
vtrace_1
wtrace_2
xtrace_3* 
6
ytrace_0
ztrace_1
{trace_2
|trace_3* 
* 
�
}iter

~beta_1

beta_2

�decay
�learning_ratem�m�%m�&m�5m�6m�?m�@m�Om�Pm�Ym�Zm�hm�im�v�v�%v�&v�5v�6v�?v�@v�Ov�Pv�Yv�Zv�hv�iv�*

�serving_default* 
 
0
1
2
3*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_15/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_15/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_15/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_15/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

%0
&1*

%0
&1*

j0
k1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_20/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
50
61
72
83*

50
61*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_16/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_16/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_16/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_16/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

?0
@1*

?0
@1*

l0
m1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_21/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
O0
P1
Q2
R3*

O0
P1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_17/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_17/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_17/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_17/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

Y0
Z1*

Y0
Z1*

n0
o1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_22/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

h0
i1*

h0
i1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_23/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 
.
0
1
72
83
Q4
R5*
J
0
1
2
3
4
5
6
7
	8

9*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

j0
k1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

70
81*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

l0
m1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Q0
R1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

n0
o1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
��
VARIABLE_VALUE#Adam/batch_normalization_15/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_15/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_20/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_20/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_16/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_16/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_21/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_21/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_17/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_17/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_22/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_22/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_23/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_23/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_15/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_15/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_20/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_20/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_16/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_16/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_21/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_21/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_17/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_17/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_22/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_22/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_23/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_23/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0batch_normalization_15/gamma/Read/ReadVariableOp/batch_normalization_15/beta/Read/ReadVariableOp6batch_normalization_15/moving_mean/Read/ReadVariableOp:batch_normalization_15/moving_variance/Read/ReadVariableOp#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOp0batch_normalization_16/gamma/Read/ReadVariableOp/batch_normalization_16/beta/Read/ReadVariableOp6batch_normalization_16/moving_mean/Read/ReadVariableOp:batch_normalization_16/moving_variance/Read/ReadVariableOp#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp0batch_normalization_17/gamma/Read/ReadVariableOp/batch_normalization_17/beta/Read/ReadVariableOp6batch_normalization_17/moving_mean/Read/ReadVariableOp:batch_normalization_17/moving_variance/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp7Adam/batch_normalization_15/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_15/beta/m/Read/ReadVariableOp*Adam/dense_20/kernel/m/Read/ReadVariableOp(Adam/dense_20/bias/m/Read/ReadVariableOp7Adam/batch_normalization_16/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_16/beta/m/Read/ReadVariableOp*Adam/dense_21/kernel/m/Read/ReadVariableOp(Adam/dense_21/bias/m/Read/ReadVariableOp7Adam/batch_normalization_17/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_17/beta/m/Read/ReadVariableOp*Adam/dense_22/kernel/m/Read/ReadVariableOp(Adam/dense_22/bias/m/Read/ReadVariableOp*Adam/dense_23/kernel/m/Read/ReadVariableOp(Adam/dense_23/bias/m/Read/ReadVariableOp7Adam/batch_normalization_15/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_15/beta/v/Read/ReadVariableOp*Adam/dense_20/kernel/v/Read/ReadVariableOp(Adam/dense_20/bias/v/Read/ReadVariableOp7Adam/batch_normalization_16/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_16/beta/v/Read/ReadVariableOp*Adam/dense_21/kernel/v/Read/ReadVariableOp(Adam/dense_21/bias/v/Read/ReadVariableOp7Adam/batch_normalization_17/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_17/beta/v/Read/ReadVariableOp*Adam/dense_22/kernel/v/Read/ReadVariableOp(Adam/dense_22/bias/v/Read/ReadVariableOp*Adam/dense_23/kernel/v/Read/ReadVariableOp(Adam/dense_23/bias/v/Read/ReadVariableOpConst*F
Tin?
=2;	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__traced_save_69853
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_variancedense_20/kerneldense_20/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_variancedense_21/kerneldense_21/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_variancedense_22/kerneldense_22/biasdense_23/kerneldense_23/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcount#Adam/batch_normalization_15/gamma/m"Adam/batch_normalization_15/beta/mAdam/dense_20/kernel/mAdam/dense_20/bias/m#Adam/batch_normalization_16/gamma/m"Adam/batch_normalization_16/beta/mAdam/dense_21/kernel/mAdam/dense_21/bias/m#Adam/batch_normalization_17/gamma/m"Adam/batch_normalization_17/beta/mAdam/dense_22/kernel/mAdam/dense_22/bias/mAdam/dense_23/kernel/mAdam/dense_23/bias/m#Adam/batch_normalization_15/gamma/v"Adam/batch_normalization_15/beta/vAdam/dense_20/kernel/vAdam/dense_20/bias/v#Adam/batch_normalization_16/gamma/v"Adam/batch_normalization_16/beta/vAdam/dense_21/kernel/vAdam/dense_21/bias/v#Adam/batch_normalization_17/gamma/v"Adam/batch_normalization_17/beta/vAdam/dense_22/kernel/vAdam/dense_22/bias/vAdam/dense_23/kernel/vAdam/dense_23/bias/v*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_restore_70034��
��
�
 __inference__wrapped_model_67773 
batch_normalization_15_inputN
@sequential_5_batch_normalization_15_cast_readvariableop_resource:)P
Bsequential_5_batch_normalization_15_cast_1_readvariableop_resource:)P
Bsequential_5_batch_normalization_15_cast_2_readvariableop_resource:)P
Bsequential_5_batch_normalization_15_cast_3_readvariableop_resource:)F
4sequential_5_dense_20_matmul_readvariableop_resource:)@C
5sequential_5_dense_20_biasadd_readvariableop_resource:@N
@sequential_5_batch_normalization_16_cast_readvariableop_resource:@P
Bsequential_5_batch_normalization_16_cast_1_readvariableop_resource:@P
Bsequential_5_batch_normalization_16_cast_2_readvariableop_resource:@P
Bsequential_5_batch_normalization_16_cast_3_readvariableop_resource:@F
4sequential_5_dense_21_matmul_readvariableop_resource:@@C
5sequential_5_dense_21_biasadd_readvariableop_resource:@N
@sequential_5_batch_normalization_17_cast_readvariableop_resource:@P
Bsequential_5_batch_normalization_17_cast_1_readvariableop_resource:@P
Bsequential_5_batch_normalization_17_cast_2_readvariableop_resource:@P
Bsequential_5_batch_normalization_17_cast_3_readvariableop_resource:@F
4sequential_5_dense_22_matmul_readvariableop_resource:@@C
5sequential_5_dense_22_biasadd_readvariableop_resource:@F
4sequential_5_dense_23_matmul_readvariableop_resource:@C
5sequential_5_dense_23_biasadd_readvariableop_resource:
identity��7sequential_5/batch_normalization_15/Cast/ReadVariableOp�9sequential_5/batch_normalization_15/Cast_1/ReadVariableOp�9sequential_5/batch_normalization_15/Cast_2/ReadVariableOp�9sequential_5/batch_normalization_15/Cast_3/ReadVariableOp�7sequential_5/batch_normalization_16/Cast/ReadVariableOp�9sequential_5/batch_normalization_16/Cast_1/ReadVariableOp�9sequential_5/batch_normalization_16/Cast_2/ReadVariableOp�9sequential_5/batch_normalization_16/Cast_3/ReadVariableOp�7sequential_5/batch_normalization_17/Cast/ReadVariableOp�9sequential_5/batch_normalization_17/Cast_1/ReadVariableOp�9sequential_5/batch_normalization_17/Cast_2/ReadVariableOp�9sequential_5/batch_normalization_17/Cast_3/ReadVariableOp�,sequential_5/dense_20/BiasAdd/ReadVariableOp�+sequential_5/dense_20/MatMul/ReadVariableOp�,sequential_5/dense_21/BiasAdd/ReadVariableOp�+sequential_5/dense_21/MatMul/ReadVariableOp�,sequential_5/dense_22/BiasAdd/ReadVariableOp�+sequential_5/dense_22/MatMul/ReadVariableOp�,sequential_5/dense_23/BiasAdd/ReadVariableOp�+sequential_5/dense_23/MatMul/ReadVariableOp�
7sequential_5/batch_normalization_15/Cast/ReadVariableOpReadVariableOp@sequential_5_batch_normalization_15_cast_readvariableop_resource*
_output_shapes
:)*
dtype0�
9sequential_5/batch_normalization_15/Cast_1/ReadVariableOpReadVariableOpBsequential_5_batch_normalization_15_cast_1_readvariableop_resource*
_output_shapes
:)*
dtype0�
9sequential_5/batch_normalization_15/Cast_2/ReadVariableOpReadVariableOpBsequential_5_batch_normalization_15_cast_2_readvariableop_resource*
_output_shapes
:)*
dtype0�
9sequential_5/batch_normalization_15/Cast_3/ReadVariableOpReadVariableOpBsequential_5_batch_normalization_15_cast_3_readvariableop_resource*
_output_shapes
:)*
dtype0x
3sequential_5/batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1sequential_5/batch_normalization_15/batchnorm/addAddV2Asequential_5/batch_normalization_15/Cast_1/ReadVariableOp:value:0<sequential_5/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:)�
3sequential_5/batch_normalization_15/batchnorm/RsqrtRsqrt5sequential_5/batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:)�
1sequential_5/batch_normalization_15/batchnorm/mulMul7sequential_5/batch_normalization_15/batchnorm/Rsqrt:y:0Asequential_5/batch_normalization_15/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:)�
3sequential_5/batch_normalization_15/batchnorm/mul_1Mulbatch_normalization_15_input5sequential_5/batch_normalization_15/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������)�
3sequential_5/batch_normalization_15/batchnorm/mul_2Mul?sequential_5/batch_normalization_15/Cast/ReadVariableOp:value:05sequential_5/batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:)�
1sequential_5/batch_normalization_15/batchnorm/subSubAsequential_5/batch_normalization_15/Cast_2/ReadVariableOp:value:07sequential_5/batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:)�
3sequential_5/batch_normalization_15/batchnorm/add_1AddV27sequential_5/batch_normalization_15/batchnorm/mul_1:z:05sequential_5/batch_normalization_15/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������)�
+sequential_5/dense_20/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_20_matmul_readvariableop_resource*
_output_shapes

:)@*
dtype0�
sequential_5/dense_20/MatMulMatMul7sequential_5/batch_normalization_15/batchnorm/add_1:z:03sequential_5/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,sequential_5/dense_20/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_5/dense_20/BiasAddBiasAdd&sequential_5/dense_20/MatMul:product:04sequential_5/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
sequential_5/dense_20/ReluRelu&sequential_5/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
 sequential_5/dropout_15/IdentityIdentity(sequential_5/dense_20/Relu:activations:0*
T0*'
_output_shapes
:���������@�
7sequential_5/batch_normalization_16/Cast/ReadVariableOpReadVariableOp@sequential_5_batch_normalization_16_cast_readvariableop_resource*
_output_shapes
:@*
dtype0�
9sequential_5/batch_normalization_16/Cast_1/ReadVariableOpReadVariableOpBsequential_5_batch_normalization_16_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
9sequential_5/batch_normalization_16/Cast_2/ReadVariableOpReadVariableOpBsequential_5_batch_normalization_16_cast_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
9sequential_5/batch_normalization_16/Cast_3/ReadVariableOpReadVariableOpBsequential_5_batch_normalization_16_cast_3_readvariableop_resource*
_output_shapes
:@*
dtype0x
3sequential_5/batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1sequential_5/batch_normalization_16/batchnorm/addAddV2Asequential_5/batch_normalization_16/Cast_1/ReadVariableOp:value:0<sequential_5/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes
:@�
3sequential_5/batch_normalization_16/batchnorm/RsqrtRsqrt5sequential_5/batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes
:@�
1sequential_5/batch_normalization_16/batchnorm/mulMul7sequential_5/batch_normalization_16/batchnorm/Rsqrt:y:0Asequential_5/batch_normalization_16/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
3sequential_5/batch_normalization_16/batchnorm/mul_1Mul)sequential_5/dropout_15/Identity:output:05sequential_5/batch_normalization_16/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
3sequential_5/batch_normalization_16/batchnorm/mul_2Mul?sequential_5/batch_normalization_16/Cast/ReadVariableOp:value:05sequential_5/batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
1sequential_5/batch_normalization_16/batchnorm/subSubAsequential_5/batch_normalization_16/Cast_2/ReadVariableOp:value:07sequential_5/batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
3sequential_5/batch_normalization_16/batchnorm/add_1AddV27sequential_5/batch_normalization_16/batchnorm/mul_1:z:05sequential_5/batch_normalization_16/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
+sequential_5/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_21_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
sequential_5/dense_21/MatMulMatMul7sequential_5/batch_normalization_16/batchnorm/add_1:z:03sequential_5/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,sequential_5/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_5/dense_21/BiasAddBiasAdd&sequential_5/dense_21/MatMul:product:04sequential_5/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
sequential_5/dense_21/ReluRelu&sequential_5/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
 sequential_5/dropout_16/IdentityIdentity(sequential_5/dense_21/Relu:activations:0*
T0*'
_output_shapes
:���������@�
7sequential_5/batch_normalization_17/Cast/ReadVariableOpReadVariableOp@sequential_5_batch_normalization_17_cast_readvariableop_resource*
_output_shapes
:@*
dtype0�
9sequential_5/batch_normalization_17/Cast_1/ReadVariableOpReadVariableOpBsequential_5_batch_normalization_17_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
9sequential_5/batch_normalization_17/Cast_2/ReadVariableOpReadVariableOpBsequential_5_batch_normalization_17_cast_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
9sequential_5/batch_normalization_17/Cast_3/ReadVariableOpReadVariableOpBsequential_5_batch_normalization_17_cast_3_readvariableop_resource*
_output_shapes
:@*
dtype0x
3sequential_5/batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1sequential_5/batch_normalization_17/batchnorm/addAddV2Asequential_5/batch_normalization_17/Cast_1/ReadVariableOp:value:0<sequential_5/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes
:@�
3sequential_5/batch_normalization_17/batchnorm/RsqrtRsqrt5sequential_5/batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes
:@�
1sequential_5/batch_normalization_17/batchnorm/mulMul7sequential_5/batch_normalization_17/batchnorm/Rsqrt:y:0Asequential_5/batch_normalization_17/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
3sequential_5/batch_normalization_17/batchnorm/mul_1Mul)sequential_5/dropout_16/Identity:output:05sequential_5/batch_normalization_17/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
3sequential_5/batch_normalization_17/batchnorm/mul_2Mul?sequential_5/batch_normalization_17/Cast/ReadVariableOp:value:05sequential_5/batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
1sequential_5/batch_normalization_17/batchnorm/subSubAsequential_5/batch_normalization_17/Cast_2/ReadVariableOp:value:07sequential_5/batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
3sequential_5/batch_normalization_17/batchnorm/add_1AddV27sequential_5/batch_normalization_17/batchnorm/mul_1:z:05sequential_5/batch_normalization_17/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
+sequential_5/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_22_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
sequential_5/dense_22/MatMulMatMul7sequential_5/batch_normalization_17/batchnorm/add_1:z:03sequential_5/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,sequential_5/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_5/dense_22/BiasAddBiasAdd&sequential_5/dense_22/MatMul:product:04sequential_5/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
sequential_5/dense_22/ReluRelu&sequential_5/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
 sequential_5/dropout_17/IdentityIdentity(sequential_5/dense_22/Relu:activations:0*
T0*'
_output_shapes
:���������@�
+sequential_5/dense_23/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
sequential_5/dense_23/MatMulMatMul)sequential_5/dropout_17/Identity:output:03sequential_5/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,sequential_5/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_5/dense_23/BiasAddBiasAdd&sequential_5/dense_23/MatMul:product:04sequential_5/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
sequential_5/dense_23/SoftmaxSoftmax&sequential_5/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������v
IdentityIdentity'sequential_5/dense_23/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp8^sequential_5/batch_normalization_15/Cast/ReadVariableOp:^sequential_5/batch_normalization_15/Cast_1/ReadVariableOp:^sequential_5/batch_normalization_15/Cast_2/ReadVariableOp:^sequential_5/batch_normalization_15/Cast_3/ReadVariableOp8^sequential_5/batch_normalization_16/Cast/ReadVariableOp:^sequential_5/batch_normalization_16/Cast_1/ReadVariableOp:^sequential_5/batch_normalization_16/Cast_2/ReadVariableOp:^sequential_5/batch_normalization_16/Cast_3/ReadVariableOp8^sequential_5/batch_normalization_17/Cast/ReadVariableOp:^sequential_5/batch_normalization_17/Cast_1/ReadVariableOp:^sequential_5/batch_normalization_17/Cast_2/ReadVariableOp:^sequential_5/batch_normalization_17/Cast_3/ReadVariableOp-^sequential_5/dense_20/BiasAdd/ReadVariableOp,^sequential_5/dense_20/MatMul/ReadVariableOp-^sequential_5/dense_21/BiasAdd/ReadVariableOp,^sequential_5/dense_21/MatMul/ReadVariableOp-^sequential_5/dense_22/BiasAdd/ReadVariableOp,^sequential_5/dense_22/MatMul/ReadVariableOp-^sequential_5/dense_23/BiasAdd/ReadVariableOp,^sequential_5/dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������): : : : : : : : : : : : : : : : : : : : 2r
7sequential_5/batch_normalization_15/Cast/ReadVariableOp7sequential_5/batch_normalization_15/Cast/ReadVariableOp2v
9sequential_5/batch_normalization_15/Cast_1/ReadVariableOp9sequential_5/batch_normalization_15/Cast_1/ReadVariableOp2v
9sequential_5/batch_normalization_15/Cast_2/ReadVariableOp9sequential_5/batch_normalization_15/Cast_2/ReadVariableOp2v
9sequential_5/batch_normalization_15/Cast_3/ReadVariableOp9sequential_5/batch_normalization_15/Cast_3/ReadVariableOp2r
7sequential_5/batch_normalization_16/Cast/ReadVariableOp7sequential_5/batch_normalization_16/Cast/ReadVariableOp2v
9sequential_5/batch_normalization_16/Cast_1/ReadVariableOp9sequential_5/batch_normalization_16/Cast_1/ReadVariableOp2v
9sequential_5/batch_normalization_16/Cast_2/ReadVariableOp9sequential_5/batch_normalization_16/Cast_2/ReadVariableOp2v
9sequential_5/batch_normalization_16/Cast_3/ReadVariableOp9sequential_5/batch_normalization_16/Cast_3/ReadVariableOp2r
7sequential_5/batch_normalization_17/Cast/ReadVariableOp7sequential_5/batch_normalization_17/Cast/ReadVariableOp2v
9sequential_5/batch_normalization_17/Cast_1/ReadVariableOp9sequential_5/batch_normalization_17/Cast_1/ReadVariableOp2v
9sequential_5/batch_normalization_17/Cast_2/ReadVariableOp9sequential_5/batch_normalization_17/Cast_2/ReadVariableOp2v
9sequential_5/batch_normalization_17/Cast_3/ReadVariableOp9sequential_5/batch_normalization_17/Cast_3/ReadVariableOp2\
,sequential_5/dense_20/BiasAdd/ReadVariableOp,sequential_5/dense_20/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_20/MatMul/ReadVariableOp+sequential_5/dense_20/MatMul/ReadVariableOp2\
,sequential_5/dense_21/BiasAdd/ReadVariableOp,sequential_5/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_21/MatMul/ReadVariableOp+sequential_5/dense_21/MatMul/ReadVariableOp2\
,sequential_5/dense_22/BiasAdd/ReadVariableOp,sequential_5/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_22/MatMul/ReadVariableOp+sequential_5/dense_22/MatMul/ReadVariableOp2\
,sequential_5/dense_23/BiasAdd/ReadVariableOp,sequential_5/dense_23/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_23/MatMul/ReadVariableOp+sequential_5/dense_23/MatMul/ReadVariableOp:e a
'
_output_shapes
:���������)
6
_user_specified_namebatch_normalization_15_input
�
�
C__inference_dense_21_layer_call_and_return_conditional_losses_68095

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�/dense_21/bias/Regularizer/L2Loss/ReadVariableOp�1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@�
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
"dense_21/kernel/Regularizer/L2LossL2Loss9dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0+dense_21/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/dense_21/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 dense_21/bias/Regularizer/L2LossL2Loss7dense_21/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_21/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_21/bias/Regularizer/mulMul(dense_21/bias/Regularizer/mul/x:output:0)dense_21/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_21/bias/Regularizer/L2Loss/ReadVariableOp2^dense_21/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_21/bias/Regularizer/L2Loss/ReadVariableOp/dense_21/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�$
!__inference__traced_restore_70034
file_prefix;
-assignvariableop_batch_normalization_15_gamma:)<
.assignvariableop_1_batch_normalization_15_beta:)C
5assignvariableop_2_batch_normalization_15_moving_mean:)G
9assignvariableop_3_batch_normalization_15_moving_variance:)4
"assignvariableop_4_dense_20_kernel:)@.
 assignvariableop_5_dense_20_bias:@=
/assignvariableop_6_batch_normalization_16_gamma:@<
.assignvariableop_7_batch_normalization_16_beta:@C
5assignvariableop_8_batch_normalization_16_moving_mean:@G
9assignvariableop_9_batch_normalization_16_moving_variance:@5
#assignvariableop_10_dense_21_kernel:@@/
!assignvariableop_11_dense_21_bias:@>
0assignvariableop_12_batch_normalization_17_gamma:@=
/assignvariableop_13_batch_normalization_17_beta:@D
6assignvariableop_14_batch_normalization_17_moving_mean:@H
:assignvariableop_15_batch_normalization_17_moving_variance:@5
#assignvariableop_16_dense_22_kernel:@@/
!assignvariableop_17_dense_22_bias:@5
#assignvariableop_18_dense_23_kernel:@/
!assignvariableop_19_dense_23_bias:'
assignvariableop_20_adam_iter:	 )
assignvariableop_21_adam_beta_1: )
assignvariableop_22_adam_beta_2: (
assignvariableop_23_adam_decay: 0
&assignvariableop_24_adam_learning_rate: %
assignvariableop_25_total_1: %
assignvariableop_26_count_1: #
assignvariableop_27_total: #
assignvariableop_28_count: E
7assignvariableop_29_adam_batch_normalization_15_gamma_m:)D
6assignvariableop_30_adam_batch_normalization_15_beta_m:)<
*assignvariableop_31_adam_dense_20_kernel_m:)@6
(assignvariableop_32_adam_dense_20_bias_m:@E
7assignvariableop_33_adam_batch_normalization_16_gamma_m:@D
6assignvariableop_34_adam_batch_normalization_16_beta_m:@<
*assignvariableop_35_adam_dense_21_kernel_m:@@6
(assignvariableop_36_adam_dense_21_bias_m:@E
7assignvariableop_37_adam_batch_normalization_17_gamma_m:@D
6assignvariableop_38_adam_batch_normalization_17_beta_m:@<
*assignvariableop_39_adam_dense_22_kernel_m:@@6
(assignvariableop_40_adam_dense_22_bias_m:@<
*assignvariableop_41_adam_dense_23_kernel_m:@6
(assignvariableop_42_adam_dense_23_bias_m:E
7assignvariableop_43_adam_batch_normalization_15_gamma_v:)D
6assignvariableop_44_adam_batch_normalization_15_beta_v:)<
*assignvariableop_45_adam_dense_20_kernel_v:)@6
(assignvariableop_46_adam_dense_20_bias_v:@E
7assignvariableop_47_adam_batch_normalization_16_gamma_v:@D
6assignvariableop_48_adam_batch_normalization_16_beta_v:@<
*assignvariableop_49_adam_dense_21_kernel_v:@@6
(assignvariableop_50_adam_dense_21_bias_v:@E
7assignvariableop_51_adam_batch_normalization_17_gamma_v:@D
6assignvariableop_52_adam_batch_normalization_17_beta_v:@<
*assignvariableop_53_adam_dense_22_kernel_v:@@6
(assignvariableop_54_adam_dense_22_bias_v:@<
*assignvariableop_55_adam_dense_23_kernel_v:@6
(assignvariableop_56_adam_dense_23_bias_v:
identity_58��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value�B�:B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp-assignvariableop_batch_normalization_15_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp.assignvariableop_1_batch_normalization_15_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp5assignvariableop_2_batch_normalization_15_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp9assignvariableop_3_batch_normalization_15_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_20_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_20_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_16_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_16_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp5assignvariableop_8_batch_normalization_16_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp9assignvariableop_9_batch_normalization_16_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_21_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_21_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_17_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_17_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp6assignvariableop_14_batch_normalization_17_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp:assignvariableop_15_batch_normalization_17_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_22_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_22_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_23_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_23_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_beta_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_learning_rateIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp7assignvariableop_29_adam_batch_normalization_15_gamma_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_batch_normalization_15_beta_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_20_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_20_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp7assignvariableop_33_adam_batch_normalization_16_gamma_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp6assignvariableop_34_adam_batch_normalization_16_beta_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_21_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_21_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp7assignvariableop_37_adam_batch_normalization_17_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_batch_normalization_17_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_22_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_22_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_23_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_23_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp7assignvariableop_43_adam_batch_normalization_15_gamma_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp6assignvariableop_44_adam_batch_normalization_15_beta_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_20_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_20_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp7assignvariableop_47_adam_batch_normalization_16_gamma_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp6assignvariableop_48_adam_batch_normalization_16_beta_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_21_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_21_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp7assignvariableop_51_adam_batch_normalization_17_gamma_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp6assignvariableop_52_adam_batch_normalization_17_beta_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_22_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_22_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_23_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_23_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_58IdentityIdentity_57:output:0^NoOp_1*
T0*
_output_shapes
: �

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_58Identity_58:output:0*�
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�[
�
G__inference_sequential_5_layer_call_and_return_conditional_losses_68468

inputs*
batch_normalization_15_68393:)*
batch_normalization_15_68395:)*
batch_normalization_15_68397:)*
batch_normalization_15_68399:) 
dense_20_68402:)@
dense_20_68404:@*
batch_normalization_16_68408:@*
batch_normalization_16_68410:@*
batch_normalization_16_68412:@*
batch_normalization_16_68414:@ 
dense_21_68417:@@
dense_21_68419:@*
batch_normalization_17_68423:@*
batch_normalization_17_68425:@*
batch_normalization_17_68427:@*
batch_normalization_17_68429:@ 
dense_22_68432:@@
dense_22_68434:@ 
dense_23_68438:@
dense_23_68440:
identity��.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�/dense_20/bias/Regularizer/L2Loss/ReadVariableOp�1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp� dense_21/StatefulPartitionedCall�/dense_21/bias/Regularizer/L2Loss/ReadVariableOp�1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp� dense_22/StatefulPartitionedCall�/dense_22/bias/Regularizer/L2Loss/ReadVariableOp�1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp� dense_23/StatefulPartitionedCall�"dropout_15/StatefulPartitionedCall�"dropout_16/StatefulPartitionedCall�"dropout_17/StatefulPartitionedCall�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_15_68393batch_normalization_15_68395batch_normalization_15_68397batch_normalization_15_68399*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������)*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_67844�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0dense_20_68402dense_20_68404*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_68054�
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_15_layer_call_and_return_conditional_losses_68330�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall+dropout_15/StatefulPartitionedCall:output:0batch_normalization_16_68408batch_normalization_16_68410batch_normalization_16_68412batch_normalization_16_68414*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_67926�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_21_68417dense_21_68419*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_68095�
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0#^dropout_15/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_68297�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0batch_normalization_17_68423batch_normalization_17_68425batch_normalization_17_68427batch_normalization_17_68429*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_68008�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_22_68432dense_22_68434*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_68136�
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_68264�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_23_68438dense_23_68440*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_68160�
1dense_20/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_20_68402*
_output_shapes

:)@*
dtype0�
"dense_20/kernel/Regularizer/L2LossL2Loss9dense_20/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_20/kernel/Regularizer/mulMul*dense_20/kernel/Regularizer/mul/x:output:0+dense_20/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
/dense_20/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_20_68404*
_output_shapes
:@*
dtype0�
 dense_20/bias/Regularizer/L2LossL2Loss7dense_20/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_20/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_20/bias/Regularizer/mulMul(dense_20/bias/Regularizer/mul/x:output:0)dense_20/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_21_68417*
_output_shapes

:@@*
dtype0�
"dense_21/kernel/Regularizer/L2LossL2Loss9dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0+dense_21/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
/dense_21/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_21_68419*
_output_shapes
:@*
dtype0�
 dense_21/bias/Regularizer/L2LossL2Loss7dense_21/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_21/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_21/bias/Regularizer/mulMul(dense_21/bias/Regularizer/mul/x:output:0)dense_21/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_22/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_22_68432*
_output_shapes

:@@*
dtype0�
"dense_22/kernel/Regularizer/L2LossL2Loss9dense_22/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0+dense_22/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
/dense_22/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_22_68434*
_output_shapes
:@*
dtype0�
 dense_22/bias/Regularizer/L2LossL2Loss7dense_22/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_22/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_22/bias/Regularizer/mulMul(dense_22/bias/Regularizer/mul/x:output:0)dense_22/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall0^dense_20/bias/Regularizer/L2Loss/ReadVariableOp2^dense_20/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_21/StatefulPartitionedCall0^dense_21/bias/Regularizer/L2Loss/ReadVariableOp2^dense_21/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_22/StatefulPartitionedCall0^dense_22/bias/Regularizer/L2Loss/ReadVariableOp2^dense_22/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_23/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������): : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2b
/dense_20/bias/Regularizer/L2Loss/ReadVariableOp/dense_20/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2b
/dense_21/bias/Regularizer/L2Loss/ReadVariableOp/dense_21/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2b
/dense_22/bias/Regularizer/L2Loss/ReadVariableOp/dense_22/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall:O K
'
_output_shapes
:���������)
 
_user_specified_nameinputs
�
c
E__inference_dropout_15_layer_call_and_return_conditional_losses_69303

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
d
E__inference_dropout_17_layer_call_and_return_conditional_losses_68264

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
c
E__inference_dropout_16_layer_call_and_return_conditional_losses_68106

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_69361

inputs*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@,
cast_2_readvariableop_resource:@,
cast_3_readvariableop_resource:@
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
F
*__inference_dropout_16_layer_call_fn_69428

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_68106`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�[
�
G__inference_sequential_5_layer_call_and_return_conditional_losses_68712 
batch_normalization_15_input*
batch_normalization_15_68637:)*
batch_normalization_15_68639:)*
batch_normalization_15_68641:)*
batch_normalization_15_68643:) 
dense_20_68646:)@
dense_20_68648:@*
batch_normalization_16_68652:@*
batch_normalization_16_68654:@*
batch_normalization_16_68656:@*
batch_normalization_16_68658:@ 
dense_21_68661:@@
dense_21_68663:@*
batch_normalization_17_68667:@*
batch_normalization_17_68669:@*
batch_normalization_17_68671:@*
batch_normalization_17_68673:@ 
dense_22_68676:@@
dense_22_68678:@ 
dense_23_68682:@
dense_23_68684:
identity��.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�/dense_20/bias/Regularizer/L2Loss/ReadVariableOp�1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp� dense_21/StatefulPartitionedCall�/dense_21/bias/Regularizer/L2Loss/ReadVariableOp�1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp� dense_22/StatefulPartitionedCall�/dense_22/bias/Regularizer/L2Loss/ReadVariableOp�1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp� dense_23/StatefulPartitionedCall�"dropout_15/StatefulPartitionedCall�"dropout_16/StatefulPartitionedCall�"dropout_17/StatefulPartitionedCall�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_15_inputbatch_normalization_15_68637batch_normalization_15_68639batch_normalization_15_68641batch_normalization_15_68643*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������)*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_67844�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0dense_20_68646dense_20_68648*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_68054�
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_15_layer_call_and_return_conditional_losses_68330�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall+dropout_15/StatefulPartitionedCall:output:0batch_normalization_16_68652batch_normalization_16_68654batch_normalization_16_68656batch_normalization_16_68658*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_67926�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_21_68661dense_21_68663*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_68095�
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0#^dropout_15/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_68297�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0batch_normalization_17_68667batch_normalization_17_68669batch_normalization_17_68671batch_normalization_17_68673*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_68008�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_22_68676dense_22_68678*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_68136�
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_68264�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_23_68682dense_23_68684*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_68160�
1dense_20/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_20_68646*
_output_shapes

:)@*
dtype0�
"dense_20/kernel/Regularizer/L2LossL2Loss9dense_20/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_20/kernel/Regularizer/mulMul*dense_20/kernel/Regularizer/mul/x:output:0+dense_20/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
/dense_20/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_20_68648*
_output_shapes
:@*
dtype0�
 dense_20/bias/Regularizer/L2LossL2Loss7dense_20/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_20/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_20/bias/Regularizer/mulMul(dense_20/bias/Regularizer/mul/x:output:0)dense_20/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_21_68661*
_output_shapes

:@@*
dtype0�
"dense_21/kernel/Regularizer/L2LossL2Loss9dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0+dense_21/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
/dense_21/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_21_68663*
_output_shapes
:@*
dtype0�
 dense_21/bias/Regularizer/L2LossL2Loss7dense_21/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_21/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_21/bias/Regularizer/mulMul(dense_21/bias/Regularizer/mul/x:output:0)dense_21/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_22/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_22_68676*
_output_shapes

:@@*
dtype0�
"dense_22/kernel/Regularizer/L2LossL2Loss9dense_22/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0+dense_22/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
/dense_22/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_22_68678*
_output_shapes
:@*
dtype0�
 dense_22/bias/Regularizer/L2LossL2Loss7dense_22/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_22/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_22/bias/Regularizer/mulMul(dense_22/bias/Regularizer/mul/x:output:0)dense_22/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall0^dense_20/bias/Regularizer/L2Loss/ReadVariableOp2^dense_20/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_21/StatefulPartitionedCall0^dense_21/bias/Regularizer/L2Loss/ReadVariableOp2^dense_21/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_22/StatefulPartitionedCall0^dense_22/bias/Regularizer/L2Loss/ReadVariableOp2^dense_22/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_23/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������): : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2b
/dense_20/bias/Regularizer/L2Loss/ReadVariableOp/dense_20/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2b
/dense_21/bias/Regularizer/L2Loss/ReadVariableOp/dense_21/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2b
/dense_22/bias/Regularizer/L2Loss/ReadVariableOp/dense_22/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall:e a
'
_output_shapes
:���������)
6
_user_specified_namebatch_normalization_15_input
�
�
C__inference_dense_22_layer_call_and_return_conditional_losses_69558

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�/dense_22/bias/Regularizer/L2Loss/ReadVariableOp�1dense_22/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@�
1dense_22/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
"dense_22/kernel/Regularizer/L2LossL2Loss9dense_22/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0+dense_22/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/dense_22/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 dense_22/bias/Regularizer/L2LossL2Loss7dense_22/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_22/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_22/bias/Regularizer/mulMul(dense_22/bias/Regularizer/mul/x:output:0)dense_22/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_22/bias/Regularizer/L2Loss/ReadVariableOp2^dense_22/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_22/bias/Regularizer/L2Loss/ReadVariableOp/dense_22/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_4_69650L
:dense_22_kernel_regularizer_l2loss_readvariableop_resource:@@
identity��1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp�
1dense_22/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:dense_22_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:@@*
dtype0�
"dense_22/kernel/Regularizer/L2LossL2Loss9dense_22/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0+dense_22/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_22/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_22/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp
�
�
C__inference_dense_22_layer_call_and_return_conditional_losses_68136

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�/dense_22/bias/Regularizer/L2Loss/ReadVariableOp�1dense_22/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@�
1dense_22/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
"dense_22/kernel/Regularizer/L2LossL2Loss9dense_22/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0+dense_22/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/dense_22/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 dense_22/bias/Regularizer/L2LossL2Loss7dense_22/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_22/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_22/bias/Regularizer/mulMul(dense_22/bias/Regularizer/mul/x:output:0)dense_22/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_22/bias/Regularizer/L2Loss/ReadVariableOp2^dense_22/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_22/bias/Regularizer/L2Loss/ReadVariableOp/dense_22/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
C__inference_dense_23_layer_call_and_return_conditional_losses_69605

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_15_layer_call_fn_69193

inputs
unknown:)
	unknown_0:)
	unknown_1:)
	unknown_2:)
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������)*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_67797o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������)`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������): : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������)
 
_user_specified_nameinputs
�
c
E__inference_dropout_17_layer_call_and_return_conditional_losses_69573

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_67879

inputs*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@,
cast_2_readvariableop_resource:@,
cast_3_readvariableop_resource:@
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
,__inference_sequential_5_layer_call_fn_68556 
batch_normalization_15_input
unknown:)
	unknown_0:)
	unknown_1:)
	unknown_2:)
	unknown_3:)@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@@

unknown_16:@

unknown_17:@

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_68468o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������): : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
'
_output_shapes
:���������)
6
_user_specified_namebatch_normalization_15_input
�
�
(__inference_dense_22_layer_call_fn_69539

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_68136o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
F
*__inference_dropout_15_layer_call_fn_69293

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_15_layer_call_and_return_conditional_losses_68065`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
__inference_loss_fn_5_69659F
8dense_22_bias_regularizer_l2loss_readvariableop_resource:@
identity��/dense_22/bias/Regularizer/L2Loss/ReadVariableOp�
/dense_22/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp8dense_22_bias_regularizer_l2loss_readvariableop_resource*
_output_shapes
:@*
dtype0�
 dense_22/bias/Regularizer/L2LossL2Loss7dense_22/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_22/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_22/bias/Regularizer/mulMul(dense_22/bias/Regularizer/mul/x:output:0)dense_22/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentity!dense_22/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^dense_22/bias/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_22/bias/Regularizer/L2Loss/ReadVariableOp/dense_22/bias/Regularizer/L2Loss/ReadVariableOp
�
�
(__inference_dense_23_layer_call_fn_69594

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_68160o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
d
E__inference_dropout_16_layer_call_and_return_conditional_losses_69450

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_17_layer_call_fn_69463

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_67961o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
,__inference_sequential_5_layer_call_fn_68234 
batch_normalization_15_input
unknown:)
	unknown_0:)
	unknown_1:)
	unknown_2:)
	unknown_3:)@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@@

unknown_16:@

unknown_17:@

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_68191o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������): : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
'
_output_shapes
:���������)
6
_user_specified_namebatch_normalization_15_input
�
F
*__inference_dropout_17_layer_call_fn_69563

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_68147`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�W
�
G__inference_sequential_5_layer_call_and_return_conditional_losses_68634 
batch_normalization_15_input*
batch_normalization_15_68559:)*
batch_normalization_15_68561:)*
batch_normalization_15_68563:)*
batch_normalization_15_68565:) 
dense_20_68568:)@
dense_20_68570:@*
batch_normalization_16_68574:@*
batch_normalization_16_68576:@*
batch_normalization_16_68578:@*
batch_normalization_16_68580:@ 
dense_21_68583:@@
dense_21_68585:@*
batch_normalization_17_68589:@*
batch_normalization_17_68591:@*
batch_normalization_17_68593:@*
batch_normalization_17_68595:@ 
dense_22_68598:@@
dense_22_68600:@ 
dense_23_68604:@
dense_23_68606:
identity��.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�/dense_20/bias/Regularizer/L2Loss/ReadVariableOp�1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp� dense_21/StatefulPartitionedCall�/dense_21/bias/Regularizer/L2Loss/ReadVariableOp�1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp� dense_22/StatefulPartitionedCall�/dense_22/bias/Regularizer/L2Loss/ReadVariableOp�1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp� dense_23/StatefulPartitionedCall�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_15_inputbatch_normalization_15_68559batch_normalization_15_68561batch_normalization_15_68563batch_normalization_15_68565*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������)*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_67797�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0dense_20_68568dense_20_68570*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_68054�
dropout_15/PartitionedCallPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_15_layer_call_and_return_conditional_losses_68065�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall#dropout_15/PartitionedCall:output:0batch_normalization_16_68574batch_normalization_16_68576batch_normalization_16_68578batch_normalization_16_68580*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_67879�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_21_68583dense_21_68585*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_68095�
dropout_16/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_68106�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0batch_normalization_17_68589batch_normalization_17_68591batch_normalization_17_68593batch_normalization_17_68595*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_67961�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_22_68598dense_22_68600*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_68136�
dropout_17/PartitionedCallPartitionedCall)dense_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_68147�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0dense_23_68604dense_23_68606*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_68160�
1dense_20/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_20_68568*
_output_shapes

:)@*
dtype0�
"dense_20/kernel/Regularizer/L2LossL2Loss9dense_20/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_20/kernel/Regularizer/mulMul*dense_20/kernel/Regularizer/mul/x:output:0+dense_20/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
/dense_20/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_20_68570*
_output_shapes
:@*
dtype0�
 dense_20/bias/Regularizer/L2LossL2Loss7dense_20/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_20/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_20/bias/Regularizer/mulMul(dense_20/bias/Regularizer/mul/x:output:0)dense_20/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_21_68583*
_output_shapes

:@@*
dtype0�
"dense_21/kernel/Regularizer/L2LossL2Loss9dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0+dense_21/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
/dense_21/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_21_68585*
_output_shapes
:@*
dtype0�
 dense_21/bias/Regularizer/L2LossL2Loss7dense_21/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_21/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_21/bias/Regularizer/mulMul(dense_21/bias/Regularizer/mul/x:output:0)dense_21/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_22/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_22_68598*
_output_shapes

:@@*
dtype0�
"dense_22/kernel/Regularizer/L2LossL2Loss9dense_22/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0+dense_22/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
/dense_22/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_22_68600*
_output_shapes
:@*
dtype0�
 dense_22/bias/Regularizer/L2LossL2Loss7dense_22/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_22/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_22/bias/Regularizer/mulMul(dense_22/bias/Regularizer/mul/x:output:0)dense_22/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall0^dense_20/bias/Regularizer/L2Loss/ReadVariableOp2^dense_20/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_21/StatefulPartitionedCall0^dense_21/bias/Regularizer/L2Loss/ReadVariableOp2^dense_21/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_22/StatefulPartitionedCall0^dense_22/bias/Regularizer/L2Loss/ReadVariableOp2^dense_22/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������): : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2b
/dense_20/bias/Regularizer/L2Loss/ReadVariableOp/dense_20/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2b
/dense_21/bias/Regularizer/L2Loss/ReadVariableOp/dense_21/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2b
/dense_22/bias/Regularizer/L2Loss/ReadVariableOp/dense_22/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:e a
'
_output_shapes
:���������)
6
_user_specified_namebatch_normalization_15_input
�
�
(__inference_dense_21_layer_call_fn_69404

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_68095o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�$
�
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_67926

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
c
E__inference_dropout_15_layer_call_and_return_conditional_losses_68065

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_17_layer_call_fn_69476

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_68008o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_16_layer_call_fn_69328

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_67879o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_69226

inputs*
cast_readvariableop_resource:),
cast_1_readvariableop_resource:),
cast_2_readvariableop_resource:),
cast_3_readvariableop_resource:)
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:)*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:)*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:)*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:)*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:)P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:)m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:)c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������)k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:)m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:)r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������)b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������)�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������): : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������)
 
_user_specified_nameinputs
�
c
*__inference_dropout_16_layer_call_fn_69433

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_68297o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
d
E__inference_dropout_16_layer_call_and_return_conditional_losses_68297

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_67961

inputs*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@,
cast_2_readvariableop_resource:@,
cast_3_readvariableop_resource:@
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
d
E__inference_dropout_15_layer_call_and_return_conditional_losses_68330

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_67797

inputs*
cast_readvariableop_resource:),
cast_1_readvariableop_resource:),
cast_2_readvariableop_resource:),
cast_3_readvariableop_resource:)
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:)*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:)*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:)*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:)*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:)P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:)m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:)c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������)k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:)m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:)r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������)b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������)�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������): : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������)
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_2_69632L
:dense_21_kernel_regularizer_l2loss_readvariableop_resource:@@
identity��1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp�
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:dense_21_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:@@*
dtype0�
"dense_21/kernel/Regularizer/L2LossL2Loss9dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0+dense_21/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_21/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_21/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp
�	
�
__inference_loss_fn_0_69614L
:dense_20_kernel_regularizer_l2loss_readvariableop_resource:)@
identity��1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp�
1dense_20/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:dense_20_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:)@*
dtype0�
"dense_20/kernel/Regularizer/L2LossL2Loss9dense_20/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_20/kernel/Regularizer/mulMul*dense_20/kernel/Regularizer/mul/x:output:0+dense_20/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_20/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_20/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp
�
c
*__inference_dropout_17_layer_call_fn_69568

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_68264o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
c
*__inference_dropout_15_layer_call_fn_69298

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_15_layer_call_and_return_conditional_losses_68330o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
C__inference_dense_21_layer_call_and_return_conditional_losses_69423

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�/dense_21/bias/Regularizer/L2Loss/ReadVariableOp�1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@�
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
"dense_21/kernel/Regularizer/L2LossL2Loss9dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0+dense_21/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/dense_21/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 dense_21/bias/Regularizer/L2LossL2Loss7dense_21/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_21/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_21/bias/Regularizer/mulMul(dense_21/bias/Regularizer/mul/x:output:0)dense_21/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_21/bias/Regularizer/L2Loss/ReadVariableOp2^dense_21/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_21/bias/Regularizer/L2Loss/ReadVariableOp/dense_21/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�$
�
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_69260

inputs5
'assignmovingavg_readvariableop_resource:)7
)assignmovingavg_1_readvariableop_resource:)*
cast_readvariableop_resource:),
cast_1_readvariableop_resource:)
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:)*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:)�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������)l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:)*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:)*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:)*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:)*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:)x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:)�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:)*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:)~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:)�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:)*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:)*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:)P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:)m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:)c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������)h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:)k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:)r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������)b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������)�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������): : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������)
 
_user_specified_nameinputs
�
�
,__inference_sequential_5_layer_call_fn_68903

inputs
unknown:)
	unknown_0:)
	unknown_1:)
	unknown_2:)
	unknown_3:)@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@@

unknown_16:@

unknown_17:@

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_68468o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������): : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������)
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_16_layer_call_fn_69341

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_67926o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
c
E__inference_dropout_16_layer_call_and_return_conditional_losses_69438

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�s
�
__inference__traced_save_69853
file_prefix;
7savev2_batch_normalization_15_gamma_read_readvariableop:
6savev2_batch_normalization_15_beta_read_readvariableopA
=savev2_batch_normalization_15_moving_mean_read_readvariableopE
Asavev2_batch_normalization_15_moving_variance_read_readvariableop.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop;
7savev2_batch_normalization_16_gamma_read_readvariableop:
6savev2_batch_normalization_16_beta_read_readvariableopA
=savev2_batch_normalization_16_moving_mean_read_readvariableopE
Asavev2_batch_normalization_16_moving_variance_read_readvariableop.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop;
7savev2_batch_normalization_17_gamma_read_readvariableop:
6savev2_batch_normalization_17_beta_read_readvariableopA
=savev2_batch_normalization_17_moving_mean_read_readvariableopE
Asavev2_batch_normalization_17_moving_variance_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopB
>savev2_adam_batch_normalization_15_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_15_beta_m_read_readvariableop5
1savev2_adam_dense_20_kernel_m_read_readvariableop3
/savev2_adam_dense_20_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_16_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_16_beta_m_read_readvariableop5
1savev2_adam_dense_21_kernel_m_read_readvariableop3
/savev2_adam_dense_21_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_17_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_17_beta_m_read_readvariableop5
1savev2_adam_dense_22_kernel_m_read_readvariableop3
/savev2_adam_dense_22_bias_m_read_readvariableop5
1savev2_adam_dense_23_kernel_m_read_readvariableop3
/savev2_adam_dense_23_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_15_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_15_beta_v_read_readvariableop5
1savev2_adam_dense_20_kernel_v_read_readvariableop3
/savev2_adam_dense_20_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_16_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_16_beta_v_read_readvariableop5
1savev2_adam_dense_21_kernel_v_read_readvariableop3
/savev2_adam_dense_21_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_17_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_17_beta_v_read_readvariableop5
1savev2_adam_dense_22_kernel_v_read_readvariableop3
/savev2_adam_dense_22_bias_v_read_readvariableop5
1savev2_adam_dense_23_kernel_v_read_readvariableop3
/savev2_adam_dense_23_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value�B�:B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_batch_normalization_15_gamma_read_readvariableop6savev2_batch_normalization_15_beta_read_readvariableop=savev2_batch_normalization_15_moving_mean_read_readvariableopAsavev2_batch_normalization_15_moving_variance_read_readvariableop*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop7savev2_batch_normalization_16_gamma_read_readvariableop6savev2_batch_normalization_16_beta_read_readvariableop=savev2_batch_normalization_16_moving_mean_read_readvariableopAsavev2_batch_normalization_16_moving_variance_read_readvariableop*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop7savev2_batch_normalization_17_gamma_read_readvariableop6savev2_batch_normalization_17_beta_read_readvariableop=savev2_batch_normalization_17_moving_mean_read_readvariableopAsavev2_batch_normalization_17_moving_variance_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop>savev2_adam_batch_normalization_15_gamma_m_read_readvariableop=savev2_adam_batch_normalization_15_beta_m_read_readvariableop1savev2_adam_dense_20_kernel_m_read_readvariableop/savev2_adam_dense_20_bias_m_read_readvariableop>savev2_adam_batch_normalization_16_gamma_m_read_readvariableop=savev2_adam_batch_normalization_16_beta_m_read_readvariableop1savev2_adam_dense_21_kernel_m_read_readvariableop/savev2_adam_dense_21_bias_m_read_readvariableop>savev2_adam_batch_normalization_17_gamma_m_read_readvariableop=savev2_adam_batch_normalization_17_beta_m_read_readvariableop1savev2_adam_dense_22_kernel_m_read_readvariableop/savev2_adam_dense_22_bias_m_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableop>savev2_adam_batch_normalization_15_gamma_v_read_readvariableop=savev2_adam_batch_normalization_15_beta_v_read_readvariableop1savev2_adam_dense_20_kernel_v_read_readvariableop/savev2_adam_dense_20_bias_v_read_readvariableop>savev2_adam_batch_normalization_16_gamma_v_read_readvariableop=savev2_adam_batch_normalization_16_beta_v_read_readvariableop1savev2_adam_dense_21_kernel_v_read_readvariableop/savev2_adam_dense_21_bias_v_read_readvariableop>savev2_adam_batch_normalization_17_gamma_v_read_readvariableop=savev2_adam_batch_normalization_17_beta_v_read_readvariableop1savev2_adam_dense_22_kernel_v_read_readvariableop/savev2_adam_dense_22_bias_v_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *H
dtypes>
<2:	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :):):):):)@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:: : : : : : : : : :):):)@:@:@:@:@@:@:@:@:@@:@:@::):):)@:@:@:@:@@:@:@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:): 

_output_shapes
:): 

_output_shapes
:): 

_output_shapes
:):$ 

_output_shapes

:)@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:): 

_output_shapes
:):$  

_output_shapes

:)@: !

_output_shapes
:@: "

_output_shapes
:@: #

_output_shapes
:@:$$ 

_output_shapes

:@@: %

_output_shapes
:@: &

_output_shapes
:@: '

_output_shapes
:@:$( 

_output_shapes

:@@: )

_output_shapes
:@:$* 

_output_shapes

:@: +

_output_shapes
:: ,

_output_shapes
:): -

_output_shapes
:):$. 

_output_shapes

:)@: /

_output_shapes
:@: 0

_output_shapes
:@: 1

_output_shapes
:@:$2 

_output_shapes

:@@: 3

_output_shapes
:@: 4

_output_shapes
:@: 5

_output_shapes
:@:$6 

_output_shapes

:@@: 7

_output_shapes
:@:$8 

_output_shapes

:@: 9

_output_shapes
:::

_output_shapes
: 
�
�
C__inference_dense_20_layer_call_and_return_conditional_losses_68054

inputs0
matmul_readvariableop_resource:)@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�/dense_20/bias/Regularizer/L2Loss/ReadVariableOp�1dense_20/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:)@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@�
1dense_20/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:)@*
dtype0�
"dense_20/kernel/Regularizer/L2LossL2Loss9dense_20/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_20/kernel/Regularizer/mulMul*dense_20/kernel/Regularizer/mul/x:output:0+dense_20/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/dense_20/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 dense_20/bias/Regularizer/L2LossL2Loss7dense_20/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_20/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_20/bias/Regularizer/mulMul(dense_20/bias/Regularizer/mul/x:output:0)dense_20/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_20/bias/Regularizer/L2Loss/ReadVariableOp2^dense_20/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������): : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_20/bias/Regularizer/L2Loss/ReadVariableOp/dense_20/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������)
 
_user_specified_nameinputs
�$
�
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_69395

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�
G__inference_sequential_5_layer_call_and_return_conditional_losses_69180

inputsL
>batch_normalization_15_assignmovingavg_readvariableop_resource:)N
@batch_normalization_15_assignmovingavg_1_readvariableop_resource:)A
3batch_normalization_15_cast_readvariableop_resource:)C
5batch_normalization_15_cast_1_readvariableop_resource:)9
'dense_20_matmul_readvariableop_resource:)@6
(dense_20_biasadd_readvariableop_resource:@L
>batch_normalization_16_assignmovingavg_readvariableop_resource:@N
@batch_normalization_16_assignmovingavg_1_readvariableop_resource:@A
3batch_normalization_16_cast_readvariableop_resource:@C
5batch_normalization_16_cast_1_readvariableop_resource:@9
'dense_21_matmul_readvariableop_resource:@@6
(dense_21_biasadd_readvariableop_resource:@L
>batch_normalization_17_assignmovingavg_readvariableop_resource:@N
@batch_normalization_17_assignmovingavg_1_readvariableop_resource:@A
3batch_normalization_17_cast_readvariableop_resource:@C
5batch_normalization_17_cast_1_readvariableop_resource:@9
'dense_22_matmul_readvariableop_resource:@@6
(dense_22_biasadd_readvariableop_resource:@9
'dense_23_matmul_readvariableop_resource:@6
(dense_23_biasadd_readvariableop_resource:
identity��&batch_normalization_15/AssignMovingAvg�5batch_normalization_15/AssignMovingAvg/ReadVariableOp�(batch_normalization_15/AssignMovingAvg_1�7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_15/Cast/ReadVariableOp�,batch_normalization_15/Cast_1/ReadVariableOp�&batch_normalization_16/AssignMovingAvg�5batch_normalization_16/AssignMovingAvg/ReadVariableOp�(batch_normalization_16/AssignMovingAvg_1�7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_16/Cast/ReadVariableOp�,batch_normalization_16/Cast_1/ReadVariableOp�&batch_normalization_17/AssignMovingAvg�5batch_normalization_17/AssignMovingAvg/ReadVariableOp�(batch_normalization_17/AssignMovingAvg_1�7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_17/Cast/ReadVariableOp�,batch_normalization_17/Cast_1/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�/dense_20/bias/Regularizer/L2Loss/ReadVariableOp�1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�/dense_21/bias/Regularizer/L2Loss/ReadVariableOp�1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�/dense_22/bias/Regularizer/L2Loss/ReadVariableOp�1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp
5batch_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_15/moments/meanMeaninputs>batch_normalization_15/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:)*
	keep_dims(�
+batch_normalization_15/moments/StopGradientStopGradient,batch_normalization_15/moments/mean:output:0*
T0*
_output_shapes

:)�
0batch_normalization_15/moments/SquaredDifferenceSquaredDifferenceinputs4batch_normalization_15/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������)�
9batch_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_15/moments/varianceMean4batch_normalization_15/moments/SquaredDifference:z:0Bbatch_normalization_15/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:)*
	keep_dims(�
&batch_normalization_15/moments/SqueezeSqueeze,batch_normalization_15/moments/mean:output:0*
T0*
_output_shapes
:)*
squeeze_dims
 �
(batch_normalization_15/moments/Squeeze_1Squeeze0batch_normalization_15/moments/variance:output:0*
T0*
_output_shapes
:)*
squeeze_dims
 q
,batch_normalization_15/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_15/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_15_assignmovingavg_readvariableop_resource*
_output_shapes
:)*
dtype0�
*batch_normalization_15/AssignMovingAvg/subSub=batch_normalization_15/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_15/moments/Squeeze:output:0*
T0*
_output_shapes
:)�
*batch_normalization_15/AssignMovingAvg/mulMul.batch_normalization_15/AssignMovingAvg/sub:z:05batch_normalization_15/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:)�
&batch_normalization_15/AssignMovingAvgAssignSubVariableOp>batch_normalization_15_assignmovingavg_readvariableop_resource.batch_normalization_15/AssignMovingAvg/mul:z:06^batch_normalization_15/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_15/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_15_assignmovingavg_1_readvariableop_resource*
_output_shapes
:)*
dtype0�
,batch_normalization_15/AssignMovingAvg_1/subSub?batch_normalization_15/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_15/moments/Squeeze_1:output:0*
T0*
_output_shapes
:)�
,batch_normalization_15/AssignMovingAvg_1/mulMul0batch_normalization_15/AssignMovingAvg_1/sub:z:07batch_normalization_15/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:)�
(batch_normalization_15/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_15_assignmovingavg_1_readvariableop_resource0batch_normalization_15/AssignMovingAvg_1/mul:z:08^batch_normalization_15/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
*batch_normalization_15/Cast/ReadVariableOpReadVariableOp3batch_normalization_15_cast_readvariableop_resource*
_output_shapes
:)*
dtype0�
,batch_normalization_15/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_15_cast_1_readvariableop_resource*
_output_shapes
:)*
dtype0k
&batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_15/batchnorm/addAddV21batch_normalization_15/moments/Squeeze_1:output:0/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:)~
&batch_normalization_15/batchnorm/RsqrtRsqrt(batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:)�
$batch_normalization_15/batchnorm/mulMul*batch_normalization_15/batchnorm/Rsqrt:y:04batch_normalization_15/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:)�
&batch_normalization_15/batchnorm/mul_1Mulinputs(batch_normalization_15/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������)�
&batch_normalization_15/batchnorm/mul_2Mul/batch_normalization_15/moments/Squeeze:output:0(batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:)�
$batch_normalization_15/batchnorm/subSub2batch_normalization_15/Cast/ReadVariableOp:value:0*batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:)�
&batch_normalization_15/batchnorm/add_1AddV2*batch_normalization_15/batchnorm/mul_1:z:0(batch_normalization_15/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������)�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:)@*
dtype0�
dense_20/MatMulMatMul*batch_normalization_15/batchnorm/add_1:z:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������@]
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?�
dropout_15/dropout/MulMuldense_20/Relu:activations:0!dropout_15/dropout/Const:output:0*
T0*'
_output_shapes
:���������@c
dropout_15/dropout/ShapeShapedense_20/Relu:activations:0*
T0*
_output_shapes
:�
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*

seedf
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@
5batch_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_16/moments/meanMeandropout_15/dropout/Mul_1:z:0>batch_normalization_16/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
+batch_normalization_16/moments/StopGradientStopGradient,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes

:@�
0batch_normalization_16/moments/SquaredDifferenceSquaredDifferencedropout_15/dropout/Mul_1:z:04batch_normalization_16/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������@�
9batch_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_16/moments/varianceMean4batch_normalization_16/moments/SquaredDifference:z:0Bbatch_normalization_16/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
&batch_normalization_16/moments/SqueezeSqueeze,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 �
(batch_normalization_16/moments/Squeeze_1Squeeze0batch_normalization_16/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 q
,batch_normalization_16/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_16/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_16_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
*batch_normalization_16/AssignMovingAvg/subSub=batch_normalization_16/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_16/moments/Squeeze:output:0*
T0*
_output_shapes
:@�
*batch_normalization_16/AssignMovingAvg/mulMul.batch_normalization_16/AssignMovingAvg/sub:z:05batch_normalization_16/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
&batch_normalization_16/AssignMovingAvgAssignSubVariableOp>batch_normalization_16_assignmovingavg_readvariableop_resource.batch_normalization_16/AssignMovingAvg/mul:z:06^batch_normalization_16/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_16/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_16_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
,batch_normalization_16/AssignMovingAvg_1/subSub?batch_normalization_16/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_16/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@�
,batch_normalization_16/AssignMovingAvg_1/mulMul0batch_normalization_16/AssignMovingAvg_1/sub:z:07batch_normalization_16/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
(batch_normalization_16/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_16_assignmovingavg_1_readvariableop_resource0batch_normalization_16/AssignMovingAvg_1/mul:z:08^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
*batch_normalization_16/Cast/ReadVariableOpReadVariableOp3batch_normalization_16_cast_readvariableop_resource*
_output_shapes
:@*
dtype0�
,batch_normalization_16/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_16_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0k
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_16/batchnorm/addAddV21batch_normalization_16/moments/Squeeze_1:output:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes
:@�
$batch_normalization_16/batchnorm/mulMul*batch_normalization_16/batchnorm/Rsqrt:y:04batch_normalization_16/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_16/batchnorm/mul_1Muldropout_15/dropout/Mul_1:z:0(batch_normalization_16/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
&batch_normalization_16/batchnorm/mul_2Mul/batch_normalization_16/moments/Squeeze:output:0(batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
$batch_normalization_16/batchnorm/subSub2batch_normalization_16/Cast/ReadVariableOp:value:0*batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_16/batchnorm/add_1AddV2*batch_normalization_16/batchnorm/mul_1:z:0(batch_normalization_16/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_21/MatMulMatMul*batch_normalization_16/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:���������@]
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?�
dropout_16/dropout/MulMuldense_21/Relu:activations:0!dropout_16/dropout/Const:output:0*
T0*'
_output_shapes
:���������@c
dropout_16/dropout/ShapeShapedense_21/Relu:activations:0*
T0*
_output_shapes
:�
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*

seed*
seed2f
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@
5batch_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_17/moments/meanMeandropout_16/dropout/Mul_1:z:0>batch_normalization_17/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
+batch_normalization_17/moments/StopGradientStopGradient,batch_normalization_17/moments/mean:output:0*
T0*
_output_shapes

:@�
0batch_normalization_17/moments/SquaredDifferenceSquaredDifferencedropout_16/dropout/Mul_1:z:04batch_normalization_17/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������@�
9batch_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_17/moments/varianceMean4batch_normalization_17/moments/SquaredDifference:z:0Bbatch_normalization_17/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
&batch_normalization_17/moments/SqueezeSqueeze,batch_normalization_17/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 �
(batch_normalization_17/moments/Squeeze_1Squeeze0batch_normalization_17/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 q
,batch_normalization_17/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_17/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_17_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
*batch_normalization_17/AssignMovingAvg/subSub=batch_normalization_17/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_17/moments/Squeeze:output:0*
T0*
_output_shapes
:@�
*batch_normalization_17/AssignMovingAvg/mulMul.batch_normalization_17/AssignMovingAvg/sub:z:05batch_normalization_17/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
&batch_normalization_17/AssignMovingAvgAssignSubVariableOp>batch_normalization_17_assignmovingavg_readvariableop_resource.batch_normalization_17/AssignMovingAvg/mul:z:06^batch_normalization_17/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_17/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_17_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
,batch_normalization_17/AssignMovingAvg_1/subSub?batch_normalization_17/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_17/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@�
,batch_normalization_17/AssignMovingAvg_1/mulMul0batch_normalization_17/AssignMovingAvg_1/sub:z:07batch_normalization_17/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
(batch_normalization_17/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_17_assignmovingavg_1_readvariableop_resource0batch_normalization_17/AssignMovingAvg_1/mul:z:08^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
*batch_normalization_17/Cast/ReadVariableOpReadVariableOp3batch_normalization_17_cast_readvariableop_resource*
_output_shapes
:@*
dtype0�
,batch_normalization_17/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_17_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0k
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_17/batchnorm/addAddV21batch_normalization_17/moments/Squeeze_1:output:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes
:@�
$batch_normalization_17/batchnorm/mulMul*batch_normalization_17/batchnorm/Rsqrt:y:04batch_normalization_17/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_17/batchnorm/mul_1Muldropout_16/dropout/Mul_1:z:0(batch_normalization_17/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
&batch_normalization_17/batchnorm/mul_2Mul/batch_normalization_17/moments/Squeeze:output:0(batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
$batch_normalization_17/batchnorm/subSub2batch_normalization_17/Cast/ReadVariableOp:value:0*batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_17/batchnorm/add_1AddV2*batch_normalization_17/batchnorm/mul_1:z:0(batch_normalization_17/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_22/MatMulMatMul*batch_normalization_17/batchnorm/add_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:���������@]
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?�
dropout_17/dropout/MulMuldense_22/Relu:activations:0!dropout_17/dropout/Const:output:0*
T0*'
_output_shapes
:���������@c
dropout_17/dropout/ShapeShapedense_22/Relu:activations:0*
T0*
_output_shapes
:�
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*

seed*
seed2f
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@�
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_23/MatMulMatMuldropout_17/dropout/Mul_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1dense_20/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:)@*
dtype0�
"dense_20/kernel/Regularizer/L2LossL2Loss9dense_20/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_20/kernel/Regularizer/mulMul*dense_20/kernel/Regularizer/mul/x:output:0+dense_20/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/dense_20/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 dense_20/bias/Regularizer/L2LossL2Loss7dense_20/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_20/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_20/bias/Regularizer/mulMul(dense_20/bias/Regularizer/mul/x:output:0)dense_20/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
"dense_21/kernel/Regularizer/L2LossL2Loss9dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0+dense_21/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/dense_21/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 dense_21/bias/Regularizer/L2LossL2Loss7dense_21/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_21/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_21/bias/Regularizer/mulMul(dense_21/bias/Regularizer/mul/x:output:0)dense_21/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_22/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
"dense_22/kernel/Regularizer/L2LossL2Loss9dense_22/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0+dense_22/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/dense_22/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 dense_22/bias/Regularizer/L2LossL2Loss7dense_22/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_22/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_22/bias/Regularizer/mulMul(dense_22/bias/Regularizer/mul/x:output:0)dense_22/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_23/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^batch_normalization_15/AssignMovingAvg6^batch_normalization_15/AssignMovingAvg/ReadVariableOp)^batch_normalization_15/AssignMovingAvg_18^batch_normalization_15/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_15/Cast/ReadVariableOp-^batch_normalization_15/Cast_1/ReadVariableOp'^batch_normalization_16/AssignMovingAvg6^batch_normalization_16/AssignMovingAvg/ReadVariableOp)^batch_normalization_16/AssignMovingAvg_18^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_16/Cast/ReadVariableOp-^batch_normalization_16/Cast_1/ReadVariableOp'^batch_normalization_17/AssignMovingAvg6^batch_normalization_17/AssignMovingAvg/ReadVariableOp)^batch_normalization_17/AssignMovingAvg_18^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_17/Cast/ReadVariableOp-^batch_normalization_17/Cast_1/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp0^dense_20/bias/Regularizer/L2Loss/ReadVariableOp2^dense_20/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp0^dense_21/bias/Regularizer/L2Loss/ReadVariableOp2^dense_21/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp0^dense_22/bias/Regularizer/L2Loss/ReadVariableOp2^dense_22/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������): : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_15/AssignMovingAvg&batch_normalization_15/AssignMovingAvg2n
5batch_normalization_15/AssignMovingAvg/ReadVariableOp5batch_normalization_15/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_15/AssignMovingAvg_1(batch_normalization_15/AssignMovingAvg_12r
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_15/Cast/ReadVariableOp*batch_normalization_15/Cast/ReadVariableOp2\
,batch_normalization_15/Cast_1/ReadVariableOp,batch_normalization_15/Cast_1/ReadVariableOp2P
&batch_normalization_16/AssignMovingAvg&batch_normalization_16/AssignMovingAvg2n
5batch_normalization_16/AssignMovingAvg/ReadVariableOp5batch_normalization_16/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_16/AssignMovingAvg_1(batch_normalization_16/AssignMovingAvg_12r
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_16/Cast/ReadVariableOp*batch_normalization_16/Cast/ReadVariableOp2\
,batch_normalization_16/Cast_1/ReadVariableOp,batch_normalization_16/Cast_1/ReadVariableOp2P
&batch_normalization_17/AssignMovingAvg&batch_normalization_17/AssignMovingAvg2n
5batch_normalization_17/AssignMovingAvg/ReadVariableOp5batch_normalization_17/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_17/AssignMovingAvg_1(batch_normalization_17/AssignMovingAvg_12r
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_17/Cast/ReadVariableOp*batch_normalization_17/Cast/ReadVariableOp2\
,batch_normalization_17/Cast_1/ReadVariableOp,batch_normalization_17/Cast_1/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2b
/dense_20/bias/Regularizer/L2Loss/ReadVariableOp/dense_20/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2b
/dense_21/bias/Regularizer/L2Loss/ReadVariableOp/dense_21/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2b
/dense_22/bias/Regularizer/L2Loss/ReadVariableOp/dense_22/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������)
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_69496

inputs*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@,
cast_2_readvariableop_resource:@,
cast_3_readvariableop_resource:@
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�$
�
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_68008

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
C__inference_dense_20_layer_call_and_return_conditional_losses_69288

inputs0
matmul_readvariableop_resource:)@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�/dense_20/bias/Regularizer/L2Loss/ReadVariableOp�1dense_20/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:)@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@�
1dense_20/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:)@*
dtype0�
"dense_20/kernel/Regularizer/L2LossL2Loss9dense_20/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_20/kernel/Regularizer/mulMul*dense_20/kernel/Regularizer/mul/x:output:0+dense_20/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/dense_20/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 dense_20/bias/Regularizer/L2LossL2Loss7dense_20/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_20/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_20/bias/Regularizer/mulMul(dense_20/bias/Regularizer/mul/x:output:0)dense_20/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_20/bias/Regularizer/L2Loss/ReadVariableOp2^dense_20/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������): : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_20/bias/Regularizer/L2Loss/ReadVariableOp/dense_20/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������)
 
_user_specified_nameinputs
�

�
C__inference_dense_23_layer_call_and_return_conditional_losses_68160

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
c
E__inference_dropout_17_layer_call_and_return_conditional_losses_68147

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_dense_20_layer_call_fn_69269

inputs
unknown:)@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_68054o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������): : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������)
 
_user_specified_nameinputs
�V
�
G__inference_sequential_5_layer_call_and_return_conditional_losses_68191

inputs*
batch_normalization_15_68026:)*
batch_normalization_15_68028:)*
batch_normalization_15_68030:)*
batch_normalization_15_68032:) 
dense_20_68055:)@
dense_20_68057:@*
batch_normalization_16_68067:@*
batch_normalization_16_68069:@*
batch_normalization_16_68071:@*
batch_normalization_16_68073:@ 
dense_21_68096:@@
dense_21_68098:@*
batch_normalization_17_68108:@*
batch_normalization_17_68110:@*
batch_normalization_17_68112:@*
batch_normalization_17_68114:@ 
dense_22_68137:@@
dense_22_68139:@ 
dense_23_68161:@
dense_23_68163:
identity��.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�/dense_20/bias/Regularizer/L2Loss/ReadVariableOp�1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp� dense_21/StatefulPartitionedCall�/dense_21/bias/Regularizer/L2Loss/ReadVariableOp�1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp� dense_22/StatefulPartitionedCall�/dense_22/bias/Regularizer/L2Loss/ReadVariableOp�1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp� dense_23/StatefulPartitionedCall�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_15_68026batch_normalization_15_68028batch_normalization_15_68030batch_normalization_15_68032*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������)*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_67797�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0dense_20_68055dense_20_68057*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_68054�
dropout_15/PartitionedCallPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_15_layer_call_and_return_conditional_losses_68065�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall#dropout_15/PartitionedCall:output:0batch_normalization_16_68067batch_normalization_16_68069batch_normalization_16_68071batch_normalization_16_68073*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_67879�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0dense_21_68096dense_21_68098*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_68095�
dropout_16/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_68106�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0batch_normalization_17_68108batch_normalization_17_68110batch_normalization_17_68112batch_normalization_17_68114*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_67961�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_22_68137dense_22_68139*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_68136�
dropout_17/PartitionedCallPartitionedCall)dense_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_68147�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0dense_23_68161dense_23_68163*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_68160�
1dense_20/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_20_68055*
_output_shapes

:)@*
dtype0�
"dense_20/kernel/Regularizer/L2LossL2Loss9dense_20/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_20/kernel/Regularizer/mulMul*dense_20/kernel/Regularizer/mul/x:output:0+dense_20/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
/dense_20/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_20_68057*
_output_shapes
:@*
dtype0�
 dense_20/bias/Regularizer/L2LossL2Loss7dense_20/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_20/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_20/bias/Regularizer/mulMul(dense_20/bias/Regularizer/mul/x:output:0)dense_20/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_21_68096*
_output_shapes

:@@*
dtype0�
"dense_21/kernel/Regularizer/L2LossL2Loss9dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0+dense_21/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
/dense_21/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_21_68098*
_output_shapes
:@*
dtype0�
 dense_21/bias/Regularizer/L2LossL2Loss7dense_21/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_21/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_21/bias/Regularizer/mulMul(dense_21/bias/Regularizer/mul/x:output:0)dense_21/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_22/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_22_68137*
_output_shapes

:@@*
dtype0�
"dense_22/kernel/Regularizer/L2LossL2Loss9dense_22/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0+dense_22/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
/dense_22/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_22_68139*
_output_shapes
:@*
dtype0�
 dense_22/bias/Regularizer/L2LossL2Loss7dense_22/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_22/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_22/bias/Regularizer/mulMul(dense_22/bias/Regularizer/mul/x:output:0)dense_22/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall0^dense_20/bias/Regularizer/L2Loss/ReadVariableOp2^dense_20/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_21/StatefulPartitionedCall0^dense_21/bias/Regularizer/L2Loss/ReadVariableOp2^dense_21/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_22/StatefulPartitionedCall0^dense_22/bias/Regularizer/L2Loss/ReadVariableOp2^dense_22/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������): : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2b
/dense_20/bias/Regularizer/L2Loss/ReadVariableOp/dense_20/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2b
/dense_21/bias/Regularizer/L2Loss/ReadVariableOp/dense_21/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2b
/dense_22/bias/Regularizer/L2Loss/ReadVariableOp/dense_22/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:O K
'
_output_shapes
:���������)
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_15_layer_call_fn_69206

inputs
unknown:)
	unknown_0:)
	unknown_1:)
	unknown_2:)
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������)*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_67844o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������)`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������): : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������)
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_69623F
8dense_20_bias_regularizer_l2loss_readvariableop_resource:@
identity��/dense_20/bias/Regularizer/L2Loss/ReadVariableOp�
/dense_20/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp8dense_20_bias_regularizer_l2loss_readvariableop_resource*
_output_shapes
:@*
dtype0�
 dense_20/bias/Regularizer/L2LossL2Loss7dense_20/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_20/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_20/bias/Regularizer/mulMul(dense_20/bias/Regularizer/mul/x:output:0)dense_20/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentity!dense_20/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^dense_20/bias/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_20/bias/Regularizer/L2Loss/ReadVariableOp/dense_20/bias/Regularizer/L2Loss/ReadVariableOp
�	
d
E__inference_dropout_15_layer_call_and_return_conditional_losses_69315

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�$
�
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_67844

inputs5
'assignmovingavg_readvariableop_resource:)7
)assignmovingavg_1_readvariableop_resource:)*
cast_readvariableop_resource:),
cast_1_readvariableop_resource:)
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:)*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:)�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������)l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:)*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:)*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:)*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:)*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:)x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:)�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:)*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:)~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:)�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:)*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:)*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:)P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:)m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:)c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������)h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:)k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:)r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������)b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������)�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������): : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������)
 
_user_specified_nameinputs
�
�
,__inference_sequential_5_layer_call_fn_68858

inputs
unknown:)
	unknown_0:)
	unknown_1:)
	unknown_2:)
	unknown_3:)@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@@

unknown_16:@

unknown_17:@

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_68191o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������): : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������)
 
_user_specified_nameinputs
�	
d
E__inference_dropout_17_layer_call_and_return_conditional_losses_69585

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�$
�
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_69530

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_68789 
batch_normalization_15_input
unknown:)
	unknown_0:)
	unknown_1:)
	unknown_2:)
	unknown_3:)@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@@

unknown_16:@

unknown_17:@

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_67773o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������): : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
'
_output_shapes
:���������)
6
_user_specified_namebatch_normalization_15_input
��
�
G__inference_sequential_5_layer_call_and_return_conditional_losses_69010

inputsA
3batch_normalization_15_cast_readvariableop_resource:)C
5batch_normalization_15_cast_1_readvariableop_resource:)C
5batch_normalization_15_cast_2_readvariableop_resource:)C
5batch_normalization_15_cast_3_readvariableop_resource:)9
'dense_20_matmul_readvariableop_resource:)@6
(dense_20_biasadd_readvariableop_resource:@A
3batch_normalization_16_cast_readvariableop_resource:@C
5batch_normalization_16_cast_1_readvariableop_resource:@C
5batch_normalization_16_cast_2_readvariableop_resource:@C
5batch_normalization_16_cast_3_readvariableop_resource:@9
'dense_21_matmul_readvariableop_resource:@@6
(dense_21_biasadd_readvariableop_resource:@A
3batch_normalization_17_cast_readvariableop_resource:@C
5batch_normalization_17_cast_1_readvariableop_resource:@C
5batch_normalization_17_cast_2_readvariableop_resource:@C
5batch_normalization_17_cast_3_readvariableop_resource:@9
'dense_22_matmul_readvariableop_resource:@@6
(dense_22_biasadd_readvariableop_resource:@9
'dense_23_matmul_readvariableop_resource:@6
(dense_23_biasadd_readvariableop_resource:
identity��*batch_normalization_15/Cast/ReadVariableOp�,batch_normalization_15/Cast_1/ReadVariableOp�,batch_normalization_15/Cast_2/ReadVariableOp�,batch_normalization_15/Cast_3/ReadVariableOp�*batch_normalization_16/Cast/ReadVariableOp�,batch_normalization_16/Cast_1/ReadVariableOp�,batch_normalization_16/Cast_2/ReadVariableOp�,batch_normalization_16/Cast_3/ReadVariableOp�*batch_normalization_17/Cast/ReadVariableOp�,batch_normalization_17/Cast_1/ReadVariableOp�,batch_normalization_17/Cast_2/ReadVariableOp�,batch_normalization_17/Cast_3/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�/dense_20/bias/Regularizer/L2Loss/ReadVariableOp�1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�/dense_21/bias/Regularizer/L2Loss/ReadVariableOp�1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�/dense_22/bias/Regularizer/L2Loss/ReadVariableOp�1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�
*batch_normalization_15/Cast/ReadVariableOpReadVariableOp3batch_normalization_15_cast_readvariableop_resource*
_output_shapes
:)*
dtype0�
,batch_normalization_15/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_15_cast_1_readvariableop_resource*
_output_shapes
:)*
dtype0�
,batch_normalization_15/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_15_cast_2_readvariableop_resource*
_output_shapes
:)*
dtype0�
,batch_normalization_15/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_15_cast_3_readvariableop_resource*
_output_shapes
:)*
dtype0k
&batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_15/batchnorm/addAddV24batch_normalization_15/Cast_1/ReadVariableOp:value:0/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:)~
&batch_normalization_15/batchnorm/RsqrtRsqrt(batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:)�
$batch_normalization_15/batchnorm/mulMul*batch_normalization_15/batchnorm/Rsqrt:y:04batch_normalization_15/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:)�
&batch_normalization_15/batchnorm/mul_1Mulinputs(batch_normalization_15/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������)�
&batch_normalization_15/batchnorm/mul_2Mul2batch_normalization_15/Cast/ReadVariableOp:value:0(batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:)�
$batch_normalization_15/batchnorm/subSub4batch_normalization_15/Cast_2/ReadVariableOp:value:0*batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:)�
&batch_normalization_15/batchnorm/add_1AddV2*batch_normalization_15/batchnorm/mul_1:z:0(batch_normalization_15/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������)�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:)@*
dtype0�
dense_20/MatMulMatMul*batch_normalization_15/batchnorm/add_1:z:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������@n
dropout_15/IdentityIdentitydense_20/Relu:activations:0*
T0*'
_output_shapes
:���������@�
*batch_normalization_16/Cast/ReadVariableOpReadVariableOp3batch_normalization_16_cast_readvariableop_resource*
_output_shapes
:@*
dtype0�
,batch_normalization_16/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_16_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
,batch_normalization_16/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_16_cast_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
,batch_normalization_16/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_16_cast_3_readvariableop_resource*
_output_shapes
:@*
dtype0k
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_16/batchnorm/addAddV24batch_normalization_16/Cast_1/ReadVariableOp:value:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes
:@�
$batch_normalization_16/batchnorm/mulMul*batch_normalization_16/batchnorm/Rsqrt:y:04batch_normalization_16/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_16/batchnorm/mul_1Muldropout_15/Identity:output:0(batch_normalization_16/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
&batch_normalization_16/batchnorm/mul_2Mul2batch_normalization_16/Cast/ReadVariableOp:value:0(batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
$batch_normalization_16/batchnorm/subSub4batch_normalization_16/Cast_2/ReadVariableOp:value:0*batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_16/batchnorm/add_1AddV2*batch_normalization_16/batchnorm/mul_1:z:0(batch_normalization_16/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_21/MatMulMatMul*batch_normalization_16/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:���������@n
dropout_16/IdentityIdentitydense_21/Relu:activations:0*
T0*'
_output_shapes
:���������@�
*batch_normalization_17/Cast/ReadVariableOpReadVariableOp3batch_normalization_17_cast_readvariableop_resource*
_output_shapes
:@*
dtype0�
,batch_normalization_17/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_17_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
,batch_normalization_17/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_17_cast_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
,batch_normalization_17/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_17_cast_3_readvariableop_resource*
_output_shapes
:@*
dtype0k
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_17/batchnorm/addAddV24batch_normalization_17/Cast_1/ReadVariableOp:value:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes
:@�
$batch_normalization_17/batchnorm/mulMul*batch_normalization_17/batchnorm/Rsqrt:y:04batch_normalization_17/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_17/batchnorm/mul_1Muldropout_16/Identity:output:0(batch_normalization_17/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
&batch_normalization_17/batchnorm/mul_2Mul2batch_normalization_17/Cast/ReadVariableOp:value:0(batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
$batch_normalization_17/batchnorm/subSub4batch_normalization_17/Cast_2/ReadVariableOp:value:0*batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_17/batchnorm/add_1AddV2*batch_normalization_17/batchnorm/mul_1:z:0(batch_normalization_17/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_22/MatMulMatMul*batch_normalization_17/batchnorm/add_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:���������@n
dropout_17/IdentityIdentitydense_22/Relu:activations:0*
T0*'
_output_shapes
:���������@�
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_23/MatMulMatMuldropout_17/Identity:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1dense_20/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:)@*
dtype0�
"dense_20/kernel/Regularizer/L2LossL2Loss9dense_20/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_20/kernel/Regularizer/mulMul*dense_20/kernel/Regularizer/mul/x:output:0+dense_20/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/dense_20/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 dense_20/bias/Regularizer/L2LossL2Loss7dense_20/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_20/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_20/bias/Regularizer/mulMul(dense_20/bias/Regularizer/mul/x:output:0)dense_20/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
"dense_21/kernel/Regularizer/L2LossL2Loss9dense_21/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_21/kernel/Regularizer/mulMul*dense_21/kernel/Regularizer/mul/x:output:0+dense_21/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/dense_21/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 dense_21/bias/Regularizer/L2LossL2Loss7dense_21/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_21/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_21/bias/Regularizer/mulMul(dense_21/bias/Regularizer/mul/x:output:0)dense_21/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_22/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
"dense_22/kernel/Regularizer/L2LossL2Loss9dense_22/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_22/kernel/Regularizer/mulMul*dense_22/kernel/Regularizer/mul/x:output:0+dense_22/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/dense_22/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 dense_22/bias/Regularizer/L2LossL2Loss7dense_22/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_22/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_22/bias/Regularizer/mulMul(dense_22/bias/Regularizer/mul/x:output:0)dense_22/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_23/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp+^batch_normalization_15/Cast/ReadVariableOp-^batch_normalization_15/Cast_1/ReadVariableOp-^batch_normalization_15/Cast_2/ReadVariableOp-^batch_normalization_15/Cast_3/ReadVariableOp+^batch_normalization_16/Cast/ReadVariableOp-^batch_normalization_16/Cast_1/ReadVariableOp-^batch_normalization_16/Cast_2/ReadVariableOp-^batch_normalization_16/Cast_3/ReadVariableOp+^batch_normalization_17/Cast/ReadVariableOp-^batch_normalization_17/Cast_1/ReadVariableOp-^batch_normalization_17/Cast_2/ReadVariableOp-^batch_normalization_17/Cast_3/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp0^dense_20/bias/Regularizer/L2Loss/ReadVariableOp2^dense_20/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp0^dense_21/bias/Regularizer/L2Loss/ReadVariableOp2^dense_21/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp0^dense_22/bias/Regularizer/L2Loss/ReadVariableOp2^dense_22/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������): : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_15/Cast/ReadVariableOp*batch_normalization_15/Cast/ReadVariableOp2\
,batch_normalization_15/Cast_1/ReadVariableOp,batch_normalization_15/Cast_1/ReadVariableOp2\
,batch_normalization_15/Cast_2/ReadVariableOp,batch_normalization_15/Cast_2/ReadVariableOp2\
,batch_normalization_15/Cast_3/ReadVariableOp,batch_normalization_15/Cast_3/ReadVariableOp2X
*batch_normalization_16/Cast/ReadVariableOp*batch_normalization_16/Cast/ReadVariableOp2\
,batch_normalization_16/Cast_1/ReadVariableOp,batch_normalization_16/Cast_1/ReadVariableOp2\
,batch_normalization_16/Cast_2/ReadVariableOp,batch_normalization_16/Cast_2/ReadVariableOp2\
,batch_normalization_16/Cast_3/ReadVariableOp,batch_normalization_16/Cast_3/ReadVariableOp2X
*batch_normalization_17/Cast/ReadVariableOp*batch_normalization_17/Cast/ReadVariableOp2\
,batch_normalization_17/Cast_1/ReadVariableOp,batch_normalization_17/Cast_1/ReadVariableOp2\
,batch_normalization_17/Cast_2/ReadVariableOp,batch_normalization_17/Cast_2/ReadVariableOp2\
,batch_normalization_17/Cast_3/ReadVariableOp,batch_normalization_17/Cast_3/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2b
/dense_20/bias/Regularizer/L2Loss/ReadVariableOp/dense_20/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp1dense_20/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2b
/dense_21/bias/Regularizer/L2Loss/ReadVariableOp/dense_21/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp1dense_21/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2b
/dense_22/bias/Regularizer/L2Loss/ReadVariableOp/dense_22/bias/Regularizer/L2Loss/ReadVariableOp2f
1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp1dense_22/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������)
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_69641F
8dense_21_bias_regularizer_l2loss_readvariableop_resource:@
identity��/dense_21/bias/Regularizer/L2Loss/ReadVariableOp�
/dense_21/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp8dense_21_bias_regularizer_l2loss_readvariableop_resource*
_output_shapes
:@*
dtype0�
 dense_21/bias/Regularizer/L2LossL2Loss7dense_21/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense_21/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_21/bias/Regularizer/mulMul(dense_21/bias/Regularizer/mul/x:output:0)dense_21/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentity!dense_21/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^dense_21/bias/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_21/bias/Regularizer/L2Loss/ReadVariableOp/dense_21/bias/Regularizer/L2Loss/ReadVariableOp"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
e
batch_normalization_15_inputE
.serving_default_batch_normalization_15_input:0���������)<
dense_230
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
axis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-_random_generator"
_tf_keras_layer
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4axis
	5gamma
6beta
7moving_mean
8moving_variance"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias"
_tf_keras_layer
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
G_random_generator"
_tf_keras_layer
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance"
_tf_keras_layer
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias"
_tf_keras_layer
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
a_random_generator"
_tf_keras_layer
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias"
_tf_keras_layer
�
0
1
2
3
%4
&5
56
67
78
89
?10
@11
O12
P13
Q14
R15
Y16
Z17
h18
i19"
trackable_list_wrapper
�
0
1
%2
&3
54
65
?6
@7
O8
P9
Y10
Z11
h12
i13"
trackable_list_wrapper
J
j0
k1
l2
m3
n4
o5"
trackable_list_wrapper
�
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
utrace_0
vtrace_1
wtrace_2
xtrace_32�
,__inference_sequential_5_layer_call_fn_68234
,__inference_sequential_5_layer_call_fn_68858
,__inference_sequential_5_layer_call_fn_68903
,__inference_sequential_5_layer_call_fn_68556�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zutrace_0zvtrace_1zwtrace_2zxtrace_3
�
ytrace_0
ztrace_1
{trace_2
|trace_32�
G__inference_sequential_5_layer_call_and_return_conditional_losses_69010
G__inference_sequential_5_layer_call_and_return_conditional_losses_69180
G__inference_sequential_5_layer_call_and_return_conditional_losses_68634
G__inference_sequential_5_layer_call_and_return_conditional_losses_68712�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zytrace_0zztrace_1z{trace_2z|trace_3
�B�
 __inference__wrapped_model_67773batch_normalization_15_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
}iter

~beta_1

beta_2

�decay
�learning_ratem�m�%m�&m�5m�6m�?m�@m�Om�Pm�Ym�Zm�hm�im�v�v�%v�&v�5v�6v�?v�@v�Ov�Pv�Yv�Zv�hv�iv�"
	optimizer
-
�serving_default"
signature_map
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_15_layer_call_fn_69193
6__inference_batch_normalization_15_layer_call_fn_69206�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_69226
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_69260�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:()2batch_normalization_15/gamma
):')2batch_normalization_15/beta
2:0) (2"batch_normalization_15/moving_mean
6:4) (2&batch_normalization_15/moving_variance
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_20_layer_call_fn_69269�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_20_layer_call_and_return_conditional_losses_69288�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:)@2dense_20/kernel
:@2dense_20/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_15_layer_call_fn_69293
*__inference_dropout_15_layer_call_fn_69298�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_15_layer_call_and_return_conditional_losses_69303
E__inference_dropout_15_layer_call_and_return_conditional_losses_69315�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
<
50
61
72
83"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_16_layer_call_fn_69328
6__inference_batch_normalization_16_layer_call_fn_69341�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_69361
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_69395�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_16/gamma
):'@2batch_normalization_16/beta
2:0@ (2"batch_normalization_16/moving_mean
6:4@ (2&batch_normalization_16/moving_variance
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_21_layer_call_fn_69404�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_21_layer_call_and_return_conditional_losses_69423�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:@@2dense_21/kernel
:@2dense_21/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_16_layer_call_fn_69428
*__inference_dropout_16_layer_call_fn_69433�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_16_layer_call_and_return_conditional_losses_69438
E__inference_dropout_16_layer_call_and_return_conditional_losses_69450�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
<
O0
P1
Q2
R3"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_17_layer_call_fn_69463
6__inference_batch_normalization_17_layer_call_fn_69476�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_69496
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_69530�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_17/gamma
):'@2batch_normalization_17/beta
2:0@ (2"batch_normalization_17/moving_mean
6:4@ (2&batch_normalization_17/moving_variance
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_22_layer_call_fn_69539�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_22_layer_call_and_return_conditional_losses_69558�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:@@2dense_22/kernel
:@2dense_22/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_17_layer_call_fn_69563
*__inference_dropout_17_layer_call_fn_69568�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_17_layer_call_and_return_conditional_losses_69573
E__inference_dropout_17_layer_call_and_return_conditional_losses_69585�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_23_layer_call_fn_69594�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_23_layer_call_and_return_conditional_losses_69605�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:@2dense_23/kernel
:2dense_23/bias
�
�trace_02�
__inference_loss_fn_0_69614�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_69623�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_69632�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_3_69641�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_4_69650�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_5_69659�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
J
0
1
72
83
Q4
R5"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_sequential_5_layer_call_fn_68234batch_normalization_15_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_5_layer_call_fn_68858inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_5_layer_call_fn_68903inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_5_layer_call_fn_68556batch_normalization_15_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_5_layer_call_and_return_conditional_losses_69010inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_5_layer_call_and_return_conditional_losses_69180inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_5_layer_call_and_return_conditional_losses_68634batch_normalization_15_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_5_layer_call_and_return_conditional_losses_68712batch_normalization_15_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
#__inference_signature_wrapper_68789batch_normalization_15_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_15_layer_call_fn_69193inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_15_layer_call_fn_69206inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_69226inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_69260inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_20_layer_call_fn_69269inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_20_layer_call_and_return_conditional_losses_69288inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dropout_15_layer_call_fn_69293inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_dropout_15_layer_call_fn_69298inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_15_layer_call_and_return_conditional_losses_69303inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_15_layer_call_and_return_conditional_losses_69315inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_16_layer_call_fn_69328inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_16_layer_call_fn_69341inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_69361inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_69395inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_21_layer_call_fn_69404inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_21_layer_call_and_return_conditional_losses_69423inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dropout_16_layer_call_fn_69428inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_dropout_16_layer_call_fn_69433inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_16_layer_call_and_return_conditional_losses_69438inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_16_layer_call_and_return_conditional_losses_69450inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_17_layer_call_fn_69463inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_17_layer_call_fn_69476inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_69496inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_69530inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_22_layer_call_fn_69539inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_22_layer_call_and_return_conditional_losses_69558inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dropout_17_layer_call_fn_69563inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_dropout_17_layer_call_fn_69568inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_17_layer_call_and_return_conditional_losses_69573inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_17_layer_call_and_return_conditional_losses_69585inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_23_layer_call_fn_69594inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_23_layer_call_and_return_conditional_losses_69605inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_69614"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_69623"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_2_69632"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_3_69641"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_4_69650"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_5_69659"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
/:-)2#Adam/batch_normalization_15/gamma/m
.:,)2"Adam/batch_normalization_15/beta/m
&:$)@2Adam/dense_20/kernel/m
 :@2Adam/dense_20/bias/m
/:-@2#Adam/batch_normalization_16/gamma/m
.:,@2"Adam/batch_normalization_16/beta/m
&:$@@2Adam/dense_21/kernel/m
 :@2Adam/dense_21/bias/m
/:-@2#Adam/batch_normalization_17/gamma/m
.:,@2"Adam/batch_normalization_17/beta/m
&:$@@2Adam/dense_22/kernel/m
 :@2Adam/dense_22/bias/m
&:$@2Adam/dense_23/kernel/m
 :2Adam/dense_23/bias/m
/:-)2#Adam/batch_normalization_15/gamma/v
.:,)2"Adam/batch_normalization_15/beta/v
&:$)@2Adam/dense_20/kernel/v
 :@2Adam/dense_20/bias/v
/:-@2#Adam/batch_normalization_16/gamma/v
.:,@2"Adam/batch_normalization_16/beta/v
&:$@@2Adam/dense_21/kernel/v
 :@2Adam/dense_21/bias/v
/:-@2#Adam/batch_normalization_17/gamma/v
.:,@2"Adam/batch_normalization_17/beta/v
&:$@@2Adam/dense_22/kernel/v
 :@2Adam/dense_22/bias/v
&:$@2Adam/dense_23/kernel/v
 :2Adam/dense_23/bias/v�
 __inference__wrapped_model_67773�%&7865?@QRPOYZhiE�B
;�8
6�3
batch_normalization_15_input���������)
� "3�0
.
dense_23"�
dense_23����������
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_69226b3�0
)�&
 �
inputs���������)
p 
� "%�"
�
0���������)
� �
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_69260b3�0
)�&
 �
inputs���������)
p
� "%�"
�
0���������)
� �
6__inference_batch_normalization_15_layer_call_fn_69193U3�0
)�&
 �
inputs���������)
p 
� "����������)�
6__inference_batch_normalization_15_layer_call_fn_69206U3�0
)�&
 �
inputs���������)
p
� "����������)�
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_69361b78653�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_69395b78653�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
6__inference_batch_normalization_16_layer_call_fn_69328U78653�0
)�&
 �
inputs���������@
p 
� "����������@�
6__inference_batch_normalization_16_layer_call_fn_69341U78653�0
)�&
 �
inputs���������@
p
� "����������@�
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_69496bQRPO3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_69530bQRPO3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
6__inference_batch_normalization_17_layer_call_fn_69463UQRPO3�0
)�&
 �
inputs���������@
p 
� "����������@�
6__inference_batch_normalization_17_layer_call_fn_69476UQRPO3�0
)�&
 �
inputs���������@
p
� "����������@�
C__inference_dense_20_layer_call_and_return_conditional_losses_69288\%&/�,
%�"
 �
inputs���������)
� "%�"
�
0���������@
� {
(__inference_dense_20_layer_call_fn_69269O%&/�,
%�"
 �
inputs���������)
� "����������@�
C__inference_dense_21_layer_call_and_return_conditional_losses_69423\?@/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� {
(__inference_dense_21_layer_call_fn_69404O?@/�,
%�"
 �
inputs���������@
� "����������@�
C__inference_dense_22_layer_call_and_return_conditional_losses_69558\YZ/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� {
(__inference_dense_22_layer_call_fn_69539OYZ/�,
%�"
 �
inputs���������@
� "����������@�
C__inference_dense_23_layer_call_and_return_conditional_losses_69605\hi/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� {
(__inference_dense_23_layer_call_fn_69594Ohi/�,
%�"
 �
inputs���������@
� "�����������
E__inference_dropout_15_layer_call_and_return_conditional_losses_69303\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
E__inference_dropout_15_layer_call_and_return_conditional_losses_69315\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� }
*__inference_dropout_15_layer_call_fn_69293O3�0
)�&
 �
inputs���������@
p 
� "����������@}
*__inference_dropout_15_layer_call_fn_69298O3�0
)�&
 �
inputs���������@
p
� "����������@�
E__inference_dropout_16_layer_call_and_return_conditional_losses_69438\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
E__inference_dropout_16_layer_call_and_return_conditional_losses_69450\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� }
*__inference_dropout_16_layer_call_fn_69428O3�0
)�&
 �
inputs���������@
p 
� "����������@}
*__inference_dropout_16_layer_call_fn_69433O3�0
)�&
 �
inputs���������@
p
� "����������@�
E__inference_dropout_17_layer_call_and_return_conditional_losses_69573\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
E__inference_dropout_17_layer_call_and_return_conditional_losses_69585\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� }
*__inference_dropout_17_layer_call_fn_69563O3�0
)�&
 �
inputs���������@
p 
� "����������@}
*__inference_dropout_17_layer_call_fn_69568O3�0
)�&
 �
inputs���������@
p
� "����������@:
__inference_loss_fn_0_69614%�

� 
� "� :
__inference_loss_fn_1_69623&�

� 
� "� :
__inference_loss_fn_2_69632?�

� 
� "� :
__inference_loss_fn_3_69641@�

� 
� "� :
__inference_loss_fn_4_69650Y�

� 
� "� :
__inference_loss_fn_5_69659Z�

� 
� "� �
G__inference_sequential_5_layer_call_and_return_conditional_losses_68634�%&7865?@QRPOYZhiM�J
C�@
6�3
batch_normalization_15_input���������)
p 

 
� "%�"
�
0���������
� �
G__inference_sequential_5_layer_call_and_return_conditional_losses_68712�%&7865?@QRPOYZhiM�J
C�@
6�3
batch_normalization_15_input���������)
p

 
� "%�"
�
0���������
� �
G__inference_sequential_5_layer_call_and_return_conditional_losses_69010v%&7865?@QRPOYZhi7�4
-�*
 �
inputs���������)
p 

 
� "%�"
�
0���������
� �
G__inference_sequential_5_layer_call_and_return_conditional_losses_69180v%&7865?@QRPOYZhi7�4
-�*
 �
inputs���������)
p

 
� "%�"
�
0���������
� �
,__inference_sequential_5_layer_call_fn_68234%&7865?@QRPOYZhiM�J
C�@
6�3
batch_normalization_15_input���������)
p 

 
� "�����������
,__inference_sequential_5_layer_call_fn_68556%&7865?@QRPOYZhiM�J
C�@
6�3
batch_normalization_15_input���������)
p

 
� "�����������
,__inference_sequential_5_layer_call_fn_68858i%&7865?@QRPOYZhi7�4
-�*
 �
inputs���������)
p 

 
� "�����������
,__inference_sequential_5_layer_call_fn_68903i%&7865?@QRPOYZhi7�4
-�*
 �
inputs���������)
p

 
� "�����������
#__inference_signature_wrapper_68789�%&7865?@QRPOYZhie�b
� 
[�X
V
batch_normalization_15_input6�3
batch_normalization_15_input���������)"3�0
.
dense_23"�
dense_23���������