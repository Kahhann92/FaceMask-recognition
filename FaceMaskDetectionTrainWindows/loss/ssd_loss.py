import tensorflow as tf

def loc_loss_helper(y_true, y_pred, y_flag, sigma=1):
    diff = tf.abs(y_true - y_pred) * tf.reduce_sum(y_flag, axis=-1, keepdims=True )
    less_than_thresh = tf.less(diff, 1 / (sigma ** 2))
    loss = tf.where(less_than_thresh, (sigma * diff)**2 * 0.5, diff - 0.5 / (sigma**2))

    y_true_number = tf.reduce_sum(y_flag)
    y_true_number_safe = tf.maximum(y_true_number, 1)

    norm_loss = tf.reduce_sum(loss) / y_true_number_safe
    return norm_loss


def cls_loss_helper(y_true, y_pred, negative_keep_ratio=3):
    y_true_number = tf.reduce_sum(y_true)
    y_true_number_safe = tf.maximum(y_true_number, 1)
    y_negative_mask = 1 - y_true
    y_negative_sum = tf.reduce_sum(y_negative_mask)
    negative_keep_numbers_float = tf.minimum(y_true_number_safe * negative_keep_ratio, y_negative_sum)
    negative_keep_numbers = tf.cast(negative_keep_numbers_float, dtype=tf.int32)

    # y_pred = tf.maximum(y_pred, 1e-15)
    # y_pred = tf.minimum(y_pred, 1 - 1e-15)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    binary_crossentropy = -(y_true * tf.log(y_pred) + (1 - y_true) * tf.log(1 - y_pred))
    positive_loss = tf.reduce_sum(y_true * binary_crossentropy)
    negative_loss = y_negative_mask * binary_crossentropy
    negative_loss_1d = tf.reshape(negative_loss, [-1])
    values, indices = tf.nn.top_k(negative_loss_1d, k=negative_keep_numbers, sorted=False)
    negative_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                  updates=tf.ones_like(indices, dtype=tf.int32),
                                  shape=tf.shape(negative_loss_1d)
                                  )
    negative_keep_mask = tf.cast(tf.reshape(negative_keep, tf.shape(y_negative_mask)), dtype=tf.float32)
    negative_loss_minied = tf.reduce_sum(negative_loss * negative_keep_mask)
    cls_loss = (positive_loss + negative_loss_minied) / y_true_number_safe
    return cls_loss


def loc_loss(y_true, y_pred):
    y_loc_true = y_true[:, :, :4]
    y_loc_mask= y_true[:, :, 4:]
    body_loc_loss_value = loc_loss_helper(y_loc_true, y_pred, y_loc_mask)
    print(body_loc_loss_value)
    return body_loc_loss_value


def cls_loss(y_true, y_pred):
    body_cls_loss_value = cls_loss_helper(y_true, y_pred, negative_keep_ratio=3)
    print(body_cls_loss_value)
    return body_cls_loss_value


