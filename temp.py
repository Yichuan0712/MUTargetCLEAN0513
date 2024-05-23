if configs.train_settings.additional_pos_weights:

    motif_logits_nucleus = motif_logits[:, 0:1, :]
    motif_logits_nucleus_export = motif_logits[:, 4:5, :]
    # Concatenate the parts that exclude indices 0 and 4
    motif_logits = torch.cat((motif_logits[:, 1:4, :], motif_logits[:, 5:, :]), dim=1)

    target_frag_nucleus = target_frag[:, 0:1, :]
    target_frag_nucleus_export = target_frag[:, 4:5, :]
    # Concatenate the parts that exclude indices 0 and 4
    target_frag = torch.cat((target_frag[:, 1:4, :], target_frag[:, 5:, :]), dim=1)

    position_loss = tools['loss_function'](motif_logits, target_frag.to(tools['train_device']))
    nucleus_position_loss = tools['loss_function_nucleus'] \
        (motif_logits_nucleus, target_frag_nucleus.to(tools['train_device']))
    nucleus_export_position_loss = tools['loss_function_nucleus_export'] \
        (motif_logits_nucleus_export, target_frag_nucleus_export.to(tools['train_device']))

else:
    position_loss = tools['loss_function'](motif_logits, target_frag.to(tools['train_device']))
# class_weights = target_frag * (tools['pos_weight'] - 1) + 1
# position_loss = torch.mean(position_loss * class_weights.to(tools['train_device']))

if configs.train_settings.data_aug.enable:
    class_loss = torch.mean(tools['loss_function_pro'](classification_head, type_protein_pt.to(
        tools['train_device'])))  # remove sample_weight_pt
else:
    class_loss = torch.mean(
        tools['loss_function_pro'](classification_head, type_protein_pt.to(tools['train_device'])) * sample_weight_pt)

if configs.train_settings.additional_pos_weights:
    train_writer.add_scalar('step class_loss', class_loss.item(), global_step=global_step)
    train_writer.add_scalar('step position_loss', position_loss.item(), global_step=global_step)
    train_writer.add_scalar('step nucleus_position_loss', nucleus_position_loss.item(),
                            global_step=global_step)
    train_writer.add_scalar('step nucleus_export_position_loss', nucleus_export_position_loss.item(),
                            global_step=global_step)
    print(f"{global_step} class_loss:{class_loss.item()}  " +
          f"position_loss:{position_loss.item()}  " +
          f"nucleus_position_loss:{nucleus_position_loss.item()}  " +
          f"nucleus_export_position_loss:{nucleus_export_position_loss.item()}")
    weighted_loss_sum = class_loss + position_loss + nucleus_position_loss + nucleus_export_position_loss
else:
    train_writer.add_scalar('step class_loss', class_loss.item(), global_step=global_step)
    train_writer.add_scalar('step position_loss', position_loss.item(), global_step=global_step)
    print(f"{global_step} class_loss:{class_loss.item()}  position_loss:{position_loss.item()}")
    weighted_loss_sum = class_loss + position_loss