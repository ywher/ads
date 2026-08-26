"""Prototype-guided ClassMix mask selection.

The v1 implementation is intentionally lightweight: it receives class scores
computed by the trainer and uses them to choose which present classes form the
ClassMix mask. It does not own any model state.
"""
import random

import torch

from lib.models.model_utils.dacs_transforms import generate_class_mask


def _squeeze_label(label):
    if label.dim() == 3 and label.shape[0] == 1:
        return label.squeeze(0)
    return label


def _present_classes(label, num_classes=None, ignore_index=255):
    label = _squeeze_label(label).detach().long()
    classes = torch.unique(label)
    classes = classes[classes != ignore_index]
    if num_classes is not None:
        classes = classes[(classes >= 0) & (classes < int(num_classes))]
    return classes


def _contains(classes, class_id):
    if class_id < 0:
        return False
    return bool((classes == int(class_id)).any().item())


def _augment_context_classes(selected, present, num_classes):
    """Add driving-scene context classes to a selected class set."""
    if selected.numel() == 0:
        return selected

    device = selected.device
    additions = []

    pole_id = 5
    light_id = 6
    sign_id = 7
    if _contains(selected, light_id) and _contains(present, pole_id):
        additions.append(pole_id)
    if _contains(selected, sign_id) and _contains(present, pole_id):
        additions.append(pole_id)
    if _contains(selected, pole_id):
        if _contains(present, light_id):
            additions.append(light_id)
        if _contains(present, sign_id):
            additions.append(sign_id)

    if int(num_classes) >= 19:
        rider_id, bicycle_id, motorcycle_id = 12, 18, 17
    elif int(num_classes) == 16:
        rider_id, bicycle_id, motorcycle_id = 11, 15, 14
    else:
        rider_id = bicycle_id = motorcycle_id = -1

    if _contains(selected, rider_id):
        if _contains(present, bicycle_id):
            additions.append(bicycle_id)
        if _contains(present, motorcycle_id):
            additions.append(motorcycle_id)

    if additions:
        extra = torch.as_tensor(additions, device=device, dtype=selected.dtype)
        selected = torch.unique(torch.cat([selected, extra]))
    else:
        selected = torch.unique(selected)
    return selected


def _choose_classes(present, class_scores, num_choose, random_prob=0.0,
                    deterministic=False):
    if num_choose <= 0 or present.numel() == 0:
        return present.new_empty((0,))
    if num_choose >= present.numel():
        return present

    if (not deterministic) and random.random() < float(random_prob):
        perm = torch.randperm(present.numel(), device=present.device)
        return present[perm[:num_choose]]

    scores = class_scores.to(device=present.device, dtype=torch.float32)
    present_scores = scores[present.long()].clamp_min(0)
    if float(present_scores.sum().detach().item()) <= 0:
        present_scores = torch.ones_like(present_scores)

    if deterministic:
        _, order = torch.sort(present_scores, descending=True)
        return present[order[:num_choose]]

    sampled = torch.multinomial(present_scores, num_choose, replacement=False)
    return present[sampled]


def _choose_classes_with_stuff_cap(
    present,
    class_scores,
    num_choose,
    stuff_set,
    stuff_max,
    random_prob=0.0,
    deterministic=False,
):
    """Select classes while limiting large-stuff choices when possible."""
    if not stuff_set or int(stuff_max) < 0:
        return _choose_classes(
            present,
            class_scores,
            num_choose,
            random_prob=random_prob,
            deterministic=deterministic,
        )
    if num_choose <= 0 or present.numel() == 0:
        return present.new_empty((0,))
    if num_choose >= present.numel():
        return present

    scores = class_scores.to(device=present.device, dtype=torch.float32)
    use_uniform = (
        (not deterministic) and random.random() < float(random_prob))
    selected = []
    remaining = present.clone()
    selected_stuff = 0

    while len(selected) < int(num_choose) and remaining.numel() > 0:
        remaining_ids = remaining.detach().cpu().tolist()
        if selected_stuff >= int(stuff_max):
            allowed = torch.as_tensor(
                [int(class_id) not in stuff_set
                 for class_id in remaining_ids],
                device=remaining.device,
                dtype=torch.bool,
            )
        else:
            allowed = torch.ones(
                remaining.numel(), device=remaining.device, dtype=torch.bool)
        candidates = remaining[allowed]
        if candidates.numel() == 0:
            break

        candidate_scores = (
            torch.ones_like(scores) if use_uniform else scores)
        chosen = _choose_classes(
            candidates,
            candidate_scores,
            1,
            random_prob=0.0,
            deterministic=deterministic,
        )
        class_id = int(chosen.item())
        selected.append(class_id)
        if class_id in stuff_set:
            selected_stuff += 1
        remaining = remaining[remaining != class_id]

    # Relax the cap only when it prevents the requested ClassMix ratio.
    if len(selected) < int(num_choose) and remaining.numel() > 0:
        fill_count = min(
            int(num_choose) - len(selected), int(remaining.numel()))
        fill_scores = torch.ones_like(scores) if use_uniform else scores
        filler = _choose_classes(
            remaining,
            fill_scores,
            fill_count,
            random_prob=0.0,
            deterministic=deterministic,
        )
        selected.extend(int(item) for item in filler.detach().cpu().tolist())

    return torch.as_tensor(
        selected[:int(num_choose)],
        device=present.device,
        dtype=present.dtype,
    )


def _valid_class_set(class_ids, num_classes, device):
    if class_ids is None:
        return set()
    if torch.is_tensor(class_ids):
        values = class_ids.detach().cpu().tolist()
    else:
        values = list(class_ids)
    return {
        int(class_id)
        for class_id in values
        if 0 <= int(class_id) < int(num_classes)
    }


def _sort_present_by_score(
    present,
    class_scores,
    descending=True,
    random_tie_break=False,
):
    if present.numel() == 0:
        return present
    scores = class_scores.to(device=present.device, dtype=torch.float32)
    if random_tie_break:
        permutation = torch.randperm(
            present.numel(), device=present.device)
        present = present[permutation]
    present_scores = scores[present.long()].clamp_min(0)
    _, order = torch.sort(
        present_scores,
        descending=descending,
        stable=bool(random_tie_break),
    )
    return present[order]


def _append_if_allowed(selected, class_id, stuff_set, stuff_max):
    class_id = int(class_id)
    if class_id in selected:
        return False
    if stuff_set and class_id in stuff_set:
        current = sum(1 for item in selected if item in stuff_set)
        if current >= int(stuff_max):
            return False
    selected.append(class_id)
    return True


def _count_in_set(selected, class_set):
    if not class_set:
        return 0
    return sum(1 for item in selected if int(item) in class_set)


def _append_with_structure_cap(selected, class_id, structure_set,
                               structure_max, enforce_cap=True):
    class_id = int(class_id)
    if class_id in selected:
        return False
    if (
        enforce_cap
        and structure_set
        and class_id in structure_set
        and int(structure_max) >= 0
        and _count_in_set(selected, structure_set) >= int(structure_max)
    ):
        return False
    selected.append(class_id)
    return True


def _zero_class_mask(label):
    if label.dim() == 2:
        return torch.zeros(
            1, 1, *label.shape, device=label.device, dtype=torch.long)
    return torch.zeros(
        1, *label.shape, device=label.device, dtype=torch.long)


def get_prototype_guided_class_masks(
    labels,
    class_ratio=0.5,
    class_scores=None,
    random_prob=0.0,
    deterministic=False,
    stuff_classes=None,
    stuff_max=1,
    apply_context=False,
    num_classes=19,
    ignore_index=255,
    return_selected=False,
):
    """Build ClassMix masks by sampling present classes with class scores.

    Args:
        labels (Tensor): [B,H,W] or [B,1,H,W] label maps.
        class_ratio (float): Fraction of present classes to select.
        class_scores (Tensor): Per-class non-negative scores [C].
        random_prob (float): Probability of falling back to uniform random
            selection for one sample.
        deterministic (bool): If true, choose top-scoring classes instead of
            sampling. Intended for tests/debugging.
        stuff_classes (Sequence[int]): Classes capped during primary selection.
        stuff_max (int): Maximum selected stuff classes before cap relaxation.
        apply_context (bool): Apply driving context class expansion.
        num_classes (int): Number of semantic classes.
        ignore_index (int): Label value ignored for class selection.
        return_selected (bool): Return final selected classes for debugging.
    """
    if labels.dim() not in (3, 4):
        raise ValueError('labels must be [B,H,W] or [B,1,H,W].')
    if labels.dim() == 4 and labels.shape[1] != 1:
        raise ValueError('4D labels must have a singleton channel dimension.')

    if class_scores is None:
        class_scores = torch.ones(num_classes, device=labels.device)
    else:
        class_scores = torch.as_tensor(class_scores, device=labels.device)
    if class_scores.numel() < int(num_classes):
        raise ValueError('class_scores must have at least num_classes values.')
    stuff_set = _valid_class_set(
        stuff_classes, num_classes, labels.device)

    class_masks = []
    num_class_choice = []
    selected_classes = []
    if class_ratio <= 0:
        for label in labels:
            class_masks.append(_zero_class_mask(label))
            num_class_choice.append(0)
            selected_classes.append(torch.empty(
                0, device=label.device, dtype=torch.long))
        if return_selected:
            return class_masks, num_class_choice, selected_classes
        return class_masks, num_class_choice

    for label in labels:
        label_2d = _squeeze_label(label)
        present = _present_classes(label_2d, num_classes, ignore_index)
        nclasses = int(present.numel())
        num_choose = int(nclasses * float(class_ratio))
        if nclasses == 0 or num_choose <= 0:
            selected = present.new_empty((0,))
            class_masks.append(_zero_class_mask(label))
            num_class_choice.append(0)
            selected_classes.append(selected)
            continue

        selected = _choose_classes_with_stuff_cap(
            present,
            class_scores[:int(num_classes)],
            num_choose,
            stuff_set=stuff_set,
            stuff_max=stuff_max,
            random_prob=random_prob,
            deterministic=deterministic,
        )
        num_class_choice.append(int(num_choose))
        if apply_context:
            selected = _augment_context_classes(selected, present, num_classes)
        else:
            selected = torch.unique(selected)
        selected_classes.append(selected)
        class_masks.append(generate_class_mask(label, selected).unsqueeze(0))

    if return_selected:
        return class_masks, num_class_choice, selected_classes
    return class_masks, num_class_choice


def get_incompatibility_veto_class_masks(
    labels,
    class_ratio=0.5,
    class_scores=None,
    apply_context=False,
    num_classes=19,
    ignore_index=255,
    return_selected=False,
):
    """Build uniform ClassMix masks after removing low-score outliers.

    The selector preserves uniform random ClassMix and only vetoes classes
    whose compatibility is below the per-image mean minus one population
    standard deviation. If too few classes remain, it falls back to the full
    present-class set so the requested mask ratio is unchanged.
    """
    if labels.dim() not in (3, 4):
        raise ValueError('labels must be [B,H,W] or [B,1,H,W].')
    if labels.dim() == 4 and labels.shape[1] != 1:
        raise ValueError('4D labels must have a singleton channel dimension.')

    if class_scores is None:
        class_scores = torch.ones(num_classes, device=labels.device)
    else:
        class_scores = torch.as_tensor(class_scores, device=labels.device)
    if class_scores.numel() < int(num_classes):
        raise ValueError('class_scores must have at least num_classes values.')
    class_scores = class_scores[:int(num_classes)].float()

    class_masks = []
    num_class_choice = []
    vetoed_class_counts = []
    selected_classes = []
    if class_ratio <= 0:
        for label in labels:
            class_masks.append(_zero_class_mask(label))
            num_class_choice.append(0)
            vetoed_class_counts.append(0)
            selected_classes.append(torch.empty(
                0, device=label.device, dtype=torch.long))
        result = (class_masks, num_class_choice, vetoed_class_counts)
        return (*result, selected_classes) if return_selected else result

    for label in labels:
        label_2d = _squeeze_label(label)
        present = _present_classes(label_2d, num_classes, ignore_index)
        nclasses = int(present.numel())
        num_choose = min(
            nclasses,
            int(nclasses * float(class_ratio)),
        )
        if nclasses == 0 or num_choose <= 0:
            selected = present.new_empty((0,))
            class_masks.append(_zero_class_mask(label))
            num_class_choice.append(0)
            vetoed_class_counts.append(0)
            selected_classes.append(selected)
            continue

        candidates = present
        vetoed_count = 0
        if nclasses >= 3:
            present_scores = class_scores[present.long()]
            threshold = (
                present_scores.mean()
                - present_scores.std(unbiased=False)
            )
            compatible = present_scores.ge(threshold)
            compatible_candidates = present[compatible]
            if compatible_candidates.numel() >= num_choose:
                candidates = compatible_candidates
                vetoed_count = nclasses - int(candidates.numel())

        permutation = torch.randperm(
            candidates.numel(),
            device=candidates.device,
        )
        selected = candidates[permutation[:num_choose]]
        if apply_context:
            selected = _augment_context_classes(
                selected, present, num_classes)
        else:
            selected = torch.unique(selected)

        class_masks.append(generate_class_mask(label, selected).unsqueeze(0))
        num_class_choice.append(num_choose)
        vetoed_class_counts.append(vetoed_count)
        selected_classes.append(selected)

    result = (class_masks, num_class_choice, vetoed_class_counts)
    return (*result, selected_classes) if return_selected else result


def get_target_deficit_quota_class_masks(
    labels,
    class_ratio=0.5,
    class_scores=None,
    quota=1,
    topk=6,
    stuff_classes=None,
    stuff_max=1,
    random_prob=0.0,
    random_tie_break=False,
    deterministic=False,
    apply_context=False,
    num_classes=19,
    ignore_index=255,
    return_selected=False,
):
    """Build ClassMix masks with an explicit high-deficit class quota.

    Compared with score-proportional sampling, this selector guarantees that
    each source-target mix includes at least ``quota`` currently present
    high-deficit source classes whenever possible. Large stuff classes can be
    capped through ``stuff_classes`` / ``stuff_max`` so the mask is not
    dominated by road/building/vegetation-like regions.
    """
    if labels.dim() not in (3, 4):
        raise ValueError('labels must be [B,H,W] or [B,1,H,W].')
    if labels.dim() == 4 and labels.shape[1] != 1:
        raise ValueError('4D labels must have a singleton channel dimension.')

    if class_scores is None:
        class_scores = torch.ones(num_classes, device=labels.device)
    else:
        class_scores = torch.as_tensor(class_scores, device=labels.device)
    if class_scores.numel() < int(num_classes):
        raise ValueError('class_scores must have at least num_classes values.')
    class_scores = class_scores[:int(num_classes)]
    stuff_set = _valid_class_set(stuff_classes, num_classes, labels.device)

    class_masks = []
    num_class_choice = []
    selected_classes = []
    if class_ratio <= 0:
        for label in labels:
            class_masks.append(_zero_class_mask(label))
            num_class_choice.append(0)
            selected_classes.append(torch.empty(
                0, device=label.device, dtype=torch.long))
        if return_selected:
            return class_masks, num_class_choice, selected_classes
        return class_masks, num_class_choice

    for label in labels:
        label_2d = _squeeze_label(label)
        present = _present_classes(label_2d, num_classes, ignore_index)
        nclasses = int(present.numel())
        num_choose = int(nclasses * float(class_ratio))
        if nclasses == 0 or num_choose <= 0:
            selected = present.new_empty((0,))
            class_masks.append(_zero_class_mask(label))
            num_class_choice.append(0)
            selected_classes.append(selected)
            continue

        num_choose = min(num_choose, nclasses)
        if (not deterministic) and random.random() < float(random_prob):
            selected = _choose_classes(
                present,
                class_scores,
                num_choose,
                random_prob=0.0,
                deterministic=False,
            )
            selected = torch.unique(selected)
        else:
            ranked = _sort_present_by_score(
                present,
                class_scores,
                descending=True,
                random_tie_break=random_tie_break,
            )
            selected_list = []
            top_count = min(max(1, int(topk)), int(ranked.numel()))
            high_deficit = ranked[:top_count]
            for class_id in high_deficit.detach().cpu().tolist():
                if len(selected_list) >= min(int(quota), num_choose):
                    break
                _append_if_allowed(
                    selected_list,
                    class_id,
                    stuff_set,
                    stuff_max,
                )

            for class_id in ranked.detach().cpu().tolist():
                if len(selected_list) >= num_choose:
                    break
                _append_if_allowed(
                    selected_list,
                    class_id,
                    stuff_set,
                    stuff_max,
                )

            # If a strict stuff cap prevents filling the mask, relax only the
            # filler stage while keeping the high-deficit quota decision.
            if len(selected_list) < num_choose:
                for class_id in ranked.detach().cpu().tolist():
                    if len(selected_list) >= num_choose:
                        break
                    if int(class_id) not in selected_list:
                        selected_list.append(int(class_id))

            selected = torch.as_tensor(
                selected_list[:num_choose],
                device=present.device,
                dtype=present.dtype,
            )

        num_class_choice.append(int(num_choose))
        if apply_context:
            selected = _augment_context_classes(selected, present, num_classes)
        else:
            selected = torch.unique(selected)
        selected_classes.append(selected)
        class_masks.append(generate_class_mask(label, selected).unsqueeze(0))

    if return_selected:
        return class_masks, num_class_choice, selected_classes
    return class_masks, num_class_choice


def get_target_need_mask_routing_class_masks(
    labels,
    class_ratio=0.5,
    class_scores=None,
    need_topk=6,
    need_min_classes=2,
    structure_classes=None,
    structure_min_classes=1,
    structure_max_classes=1,
    dynamic_classes=None,
    dynamic_min_classes=1,
    random_prob=0.0,
    deterministic=False,
    apply_context=False,
    num_classes=19,
    ignore_index=255,
    return_selected=False,
):
    """Build ClassMix masks with target-need routing constraints.

    TNMR-v2 keeps the target-deficit score design but routes the actual mask
    through three explicit constraints:

    1. include high target-need classes when they are present in the source;
    2. keep at least a small amount of structure/context classes;
    3. avoid masks being dominated by structure classes by reserving capacity
       for dynamic/small-object classes.
    """
    if labels.dim() not in (3, 4):
        raise ValueError('labels must be [B,H,W] or [B,1,H,W].')
    if labels.dim() == 4 and labels.shape[1] != 1:
        raise ValueError('4D labels must have a singleton channel dimension.')

    if class_scores is None:
        class_scores = torch.ones(num_classes, device=labels.device)
    else:
        class_scores = torch.as_tensor(class_scores, device=labels.device)
    if class_scores.numel() < int(num_classes):
        raise ValueError('class_scores must have at least num_classes values.')
    class_scores = class_scores[:int(num_classes)]

    structure_set = _valid_class_set(
        structure_classes, num_classes, labels.device)
    dynamic_set = _valid_class_set(dynamic_classes, num_classes, labels.device)
    structure_min = max(0, int(structure_min_classes))
    structure_max = int(structure_max_classes)
    if structure_max >= 0:
        structure_max = max(structure_max, min(structure_min, int(num_classes)))
    dynamic_min = max(0, int(dynamic_min_classes))
    need_min = max(0, int(need_min_classes))
    need_topk = max(1, int(need_topk))

    class_masks = []
    num_class_choice = []
    selected_classes = []
    if class_ratio <= 0:
        for label in labels:
            class_masks.append(_zero_class_mask(label))
            num_class_choice.append(0)
            selected_classes.append(torch.empty(
                0, device=label.device, dtype=torch.long))
        if return_selected:
            return class_masks, num_class_choice, selected_classes
        return class_masks, num_class_choice

    for label in labels:
        label_2d = _squeeze_label(label)
        present = _present_classes(label_2d, num_classes, ignore_index)
        nclasses = int(present.numel())
        num_choose = int(nclasses * float(class_ratio))
        if nclasses == 0 or num_choose <= 0:
            selected = present.new_empty((0,))
            class_masks.append(_zero_class_mask(label))
            num_class_choice.append(0)
            selected_classes.append(selected)
            continue

        num_choose = min(num_choose, nclasses)
        if (not deterministic) and random.random() < float(random_prob):
            selected = _choose_classes(
                present,
                class_scores,
                num_choose,
                random_prob=0.0,
                deterministic=False,
            )
            selected = torch.unique(selected)
        else:
            ranked = _sort_present_by_score(
                present,
                class_scores,
                descending=True,
            )
            ranked_list = [int(x) for x in ranked.detach().cpu().tolist()]
            selected_list = []

            if structure_set and structure_min > 0:
                for class_id in ranked_list:
                    if _count_in_set(selected_list, structure_set) >= min(
                        structure_min, num_choose
                    ):
                        break
                    if class_id in structure_set:
                        _append_with_structure_cap(
                            selected_list,
                            class_id,
                            structure_set,
                            structure_max,
                            enforce_cap=True,
                        )

            high_need = ranked_list[:min(need_topk, len(ranked_list))]
            for class_id in high_need:
                if len(selected_list) >= num_choose:
                    break
                if sum(1 for item in selected_list if item in high_need) >= min(
                    need_min, num_choose
                ):
                    break
                _append_with_structure_cap(
                    selected_list,
                    class_id,
                    structure_set,
                    structure_max,
                    enforce_cap=True,
                )

            if dynamic_set and dynamic_min > 0:
                for class_id in ranked_list:
                    if len(selected_list) >= num_choose:
                        break
                    if _count_in_set(selected_list, dynamic_set) >= min(
                        dynamic_min, num_choose
                    ):
                        break
                    if class_id in dynamic_set:
                        _append_with_structure_cap(
                            selected_list,
                            class_id,
                            structure_set,
                            structure_max,
                            enforce_cap=True,
                        )

            for class_id in ranked_list:
                if len(selected_list) >= num_choose:
                    break
                _append_with_structure_cap(
                    selected_list,
                    class_id,
                    structure_set,
                    structure_max,
                    enforce_cap=True,
                )

            # Relax the structure cap only if strict routing cannot fill the
            # requested ClassMix ratio for the current source sample.
            if len(selected_list) < num_choose:
                for class_id in ranked_list:
                    if len(selected_list) >= num_choose:
                        break
                    _append_with_structure_cap(
                        selected_list,
                        class_id,
                        structure_set,
                        structure_max,
                        enforce_cap=False,
                    )

            selected = torch.as_tensor(
                selected_list[:num_choose],
                device=present.device,
                dtype=present.dtype,
            )

        num_class_choice.append(int(num_choose))
        if apply_context:
            selected = _augment_context_classes(selected, present, num_classes)
        else:
            selected = torch.unique(selected)
        selected_classes.append(selected)
        class_masks.append(generate_class_mask(label, selected).unsqueeze(0))

    if return_selected:
        return class_masks, num_class_choice, selected_classes
    return class_masks, num_class_choice
