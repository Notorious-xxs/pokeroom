# 腾讯云托管启动入口
import os

import os
import time
import random
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================
# 加载YOLO模型
# ============================================================
MODEL_PATH = 'yolov8m_synthetic.pt'
_model = None


def get_model():
    """懒加载 YOLO 模型，避免容器启动超时"""
    global _model
    if _model is None:
        print(f"🔄 首次请求，正在加载模型: {MODEL_PATH}")
        try:
            _model = YOLO(MODEL_PATH)
            print(f"✅ 模型加载完成: {MODEL_PATH}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    return _model


# ============================================================
# 主识别函数：YOLO 识别 + 去重 + 手牌/公共牌分离
# ============================================================
def detect_cards_yolo(image_path):
    try:
        _model = get_model()
    except Exception:
        return {'community': [], 'hand': []}

    print(f"🔍 开始YOLO识别: {image_path}")
    start_time = time.time()

    try:
        results = _model(image_path, conf=0.02, iou=0.4, verbose=False)
        # results = model(
        #     image_path,
        #     conf=0.3,  # 提高到0.3，过滤低置信度
        #     iou=0.5,  # 提高NMS阈值
        #     verbose=False
        # )
        raw_detections = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = xyxy
                    area = (x2 - x1) * (y2 - y1)
                    conf = float(box.conf[0])

                    # 面积过滤
                    if area < 500:
                        continue

                    # 置信度过滤 - 低于0.3的检测直接丢弃
                    if conf < 0.3:
                        continue

                    rank = ''
                    suit_char = ''
                    for ch in class_name:
                        if ch.isdigit() or ch in 'ATQJK':
                            rank += ch
                        elif ch.isalpha():
                            suit_char = ch
                    suit_map = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
                    suit = suit_map.get(suit_char.upper(), suit_char)

                    raw_detections.append({
                        'class_name': class_name,
                        'suit': suit,
                        'rank': rank,
                        'confidence': conf,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'cx': (x1 + x2) / 2,
                        'cy': (y1 + y2) / 2,
                        'bw': x2 - x1,
                        'bh': y2 - y1,
                        'area': area,
                    })

        # ===== 更激进的去重逻辑 =====
        # 先按置信度排序，保留最高的
        raw_detections.sort(key=lambda x: x['confidence'], reverse=True)

        detections = []
        for det in raw_detections:
            is_dup = False
            for kept in detections:
                # 如果是同一张牌（相同花色和点数）
                if det['suit'] == kept['suit'] and det['rank'] == kept['rank']:
                    # 检查中心点距离，如果距离小于牌宽度的1.5倍，认为是重复
                    center_dist = ((det['cx'] - kept['cx']) ** 2 + (det['cy'] - kept['cy']) ** 2) ** 0.5
                    avg_width = (det['bw'] + kept['bw']) / 2

                    # 如果中心距离小于平均宽度的1.5倍，或者IoU > 0.1
                    if center_dist < avg_width * 1.5:
                        is_dup = True
                        # 保留置信度更高的
                        if det['confidence'] > kept['confidence']:
                            kept.update({
                                'x1': det['x1'], 'y1': det['y1'],
                                'x2': det['x2'], 'y2': det['y2'],
                                'cx': det['cx'], 'cy': det['cy'],
                                'bw': det['bw'], 'bh': det['bh'],
                                'area': det['area'],
                                'confidence': det['confidence']
                            })
                        break
            if not is_dup:
                detections.append(det)

        print(f"  YOLO原始检测: {len(raw_detections)} 张 → 去重后: {len(detections)} 张")

    except Exception as e:
        print(f"  ❌ YOLO检测异常: {e}")
        import traceback
        traceback.print_exc()
        return {'community': [], 'hand': []}

    # ===== 打印所有识别到的牌 =====
    print("\n" + "=" * 50)
    print(f"📋 识别到的所有牌（共 {len(detections)} 张）：")
    print("=" * 50)
    for i, det in enumerate(detections, 1):
        print(f"  {i}. {det['suit']}{det['rank']}  "
              f"置信度: {det['confidence']:.3f}  "
              f"位置: ({det['cx']:.0f}, {det['cy']:.0f})  "
              f"尺寸: {det['bw']:.0f}x{det['bh']:.0f}")
    print("=" * 50 + "\n")

    if len(detections) == 0:
        return {'community': [], 'hand': []}

    # 如果去重后仍然超过7张，用更暴力的方法
    if len(detections) > 7:
        print(f"  ⚠️ 去重后仍有{len(detections)}张，使用暴力去重")
        # 按花色和点数分组，每组只保留置信度最高的
        card_groups = {}
        for det in detections:
            key = f"{det['suit']}{det['rank']}"
            if key not in card_groups or det['confidence'] > card_groups[key]['confidence']:
                card_groups[key] = det
        detections = list(card_groups.values())
        print(f"  暴力去重后: {len(detections)} 张")

    # ===== 再次检查：如果有多余的低置信度牌，进行过滤 =====
    # 如果检测到的牌超过5张，但有些置信度明显较低，可能是误检
    if len(detections) > 5:
        # 按置信度排序
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        # 计算平均置信度
        avg_conf = sum(d['confidence'] for d in detections) / len(detections)
        # 如果某些牌置信度低于平均置信度的50%，可能是误检
        filtered = [d for d in detections if d['confidence'] >= avg_conf * 0.5]
        if len(filtered) >= 5:  # 至少保留5张
            detections = filtered
            print(f"  低置信度过滤后: {len(detections)} 张")

    # ===== 手牌和公共牌分离 =====
    # 按x坐标排序
    sorted_by_x = sorted(detections, key=lambda x: x['cx'])

    # 取最左边的2张牌作为手牌
    hand_indices = [detections.index(sorted_by_x[0]), detections.index(sorted_by_x[1])]

    # 其余牌作为公共牌
    community_indices = [k for k in range(len(detections)) if k not in hand_indices]

    # 检查是否有重复的牌在手牌和公共牌中
    hand_keys = [f"{detections[i]['suit']}{detections[i]['rank']}" for i in hand_indices]
    community_keys = [f"{detections[i]['suit']}{detections[i]['rank']}" for i in community_indices]

    # 如果有重复，从公共牌中移除
    duplicates = []
    for i, key in enumerate(community_keys):
        if key in hand_keys:
            duplicates.append(i)

    if duplicates:
        print(f"  ⚠️ 发现重复牌在公共牌中: {[community_keys[i] for i in duplicates]}")
        community_indices = [community_indices[i] for i in range(len(community_indices)) if i not in duplicates]
        print(f"  移除后公共牌: {len(community_indices)} 张")

    hand_cards = [detections[i] for i in hand_indices]
    community_cards = [detections[i] for i in community_indices]

    # 按x坐标排序公共牌
    community_cards.sort(key=lambda x: x['cx'])

    detected_community = [{
        'suit': det['suit'],
        'rank': det['rank'],
        'class_name': det['class_name'],
        'confidence': round(det['confidence'], 3),
        'source': 'yolo',
    } for det in community_cards]

    detected_hand = [{
        'suit': det['suit'],
        'rank': det['rank'],
        'class_name': det['class_name'],
        'confidence': round(det['confidence'], 3),
        'source': 'yolo',
    } for det in hand_cards]

    elapsed = time.time() - start_time
    print(f"✅ 识别完成: 公共牌{len(detected_community)}张, 手牌{len(detected_hand)}张, 耗时{elapsed:.2f}秒")
    for c in detected_community:
        print(f"   公共牌: {c['suit']}{c['rank']} (conf={c['confidence']})")
    for h in detected_hand:
        print(f"   手牌:   {h['suit']}{h['rank']} (conf={h['confidence']})")

    return {'community': detected_community, 'hand': detected_hand}


def calc_iou(a, b):
    """计算两张牌bbox的IoU（交并比），用于去重。"""
    inter_x1 = max(a['x1'], b['x1'])
    inter_y1 = max(a['y1'], b['y1'])
    inter_x2 = min(a['x2'], b['x2'])
    inter_y2 = min(a['y2'], b['y2'])

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (a['x2'] - a['x1']) * (a['y2'] - a['y1'])
    area_b = (b['x2'] - b['x1']) * (b['y2'] - b['y1'])
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0


# ============================================================
# 路由1: 首页
# ============================================================
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/poker')
def poker():
    return render_template('poker.html')


# ============================================================
# 路由2: 上传识别 (拍照接口)
# ============================================================
@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return '', 200

    print(f"📨 收到上传请求")

    try:
        if 'photo' not in request.files:
            return jsonify({'error': '没有图片'}), 400

        file = request.files['photo']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的图片格式'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"✅ 文件已保存: {filepath}")

        detected_cards = detect_cards_yolo(filepath)

        if detected_cards and (detected_cards.get('community') or detected_cards.get('hand')):
            return jsonify({
                'success': True,
                'community': detected_cards.get('community', []),
                'hand': detected_cards.get('hand', []),
                'community_count': len(detected_cards.get('community', [])),
                'hand_count': len(detected_cards.get('hand', [])),
                'total': len(detected_cards.get('community', [])) + len(detected_cards.get('hand', []))
            })
        else:
            return jsonify({
                'success': False,
                'message': '未检测到扑克牌'
            })

    except Exception as e:
        print(f"❌ 处理异常: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


# ============================================================
# 路由3: 胜率计算 (新增的接口)
# 注意：这个路由是写在 /upload 后面的！
# ============================================================
@app.route('/calculate', methods=['POST', 'OPTIONS'])
def calculate_win_rate():
    """
    计算胜率接口
    请求体: { "community": [{"suit":"♠","rank":"A"}, ...], "hand": [{"suit":"♥","rank":"K"}, ...] }
    返回: { "win": 4500, "lose": 4000, "tie": 1500, "total_hands": 10000 }
    """
    if request.method == 'OPTIONS':
        return '', 200

    try:
        from itertools import combinations
        data = request.get_json()
        community = data.get('community', [])
        hand = data.get('hand', [])

        print(f"📊 胜率计算: 公共牌 {len(community)} 张, 手牌 {len(hand)} 张")

        if len(hand) != 2:
            return jsonify({'success': False, 'message': '需要2张手牌'}), 400

        # ============================================================
        # 1. 牌面数值映射
        # ============================================================
        rank_values = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        suits_list = ['♠', '♥', '♦', '♣']

        def generate_deck():
            deck = []
            for suit in suits_list:
                for rank in rank_values.keys():
                    deck.append({'suit': suit, 'rank': rank})
            return deck

        # ============================================================
        # 2. 牌型评估函数 — 修复版：正确处理同花顺判断
        # ============================================================
        def evaluate_hand(cards):
            """
            返回 (牌型等级, 排序键值)
            牌型等级: 9=皇家同花顺, 8=同花顺, 7=四条, 6=葫芦, 5=同花, 4=顺子, 3=三条, 2=两对, 1=一对, 0=高牌
            """
            if len(cards) < 5:
                return (0, [])

            ranks = [rank_values[c['rank']] for c in cards]
            suits = [c['suit'] for c in cards]
            rank_counts = {}
            for r in ranks:
                rank_counts[r] = rank_counts.get(r, 0) + 1

            sorted_ranks = sorted(ranks, reverse=True)
            is_flush = len(set(suits)) == 1

            # 检查顺子
            unique_ranks = sorted(set(ranks), reverse=True)
            is_straight = False
            straight_high = 0
            if len(unique_ranks) >= 5:
                for i in range(len(unique_ranks) - 4):
                    if unique_ranks[i] - unique_ranks[i + 4] == 4:
                        is_straight = True
                        straight_high = unique_ranks[i]
                        break
                # A-2-3-4-5 特殊顺子
                if set([14, 2, 3, 4, 5]).issubset(set(ranks)):
                    is_straight = True
                    straight_high = 5

            counts = list(rank_counts.values())
            has_pair = 2 in counts
            has_three = 3 in counts
            has_four = 4 in counts
            pair_count = counts.count(2)

            # 牌型判断 — 关键修复：同花顺必须在同花色内判断
            if is_flush and is_straight:
                # 皇家同花顺: A-K-Q-J-10 同花
                if straight_high == 14:
                    return (9, sorted_ranks)  # 皇家同花顺
                return (8, [straight_high] + sorted_ranks)  # 同花顺

            if has_four:
                four_rank = [r for r, c in rank_counts.items() if c == 4][0]
                kicker = [r for r, c in rank_counts.items() if c == 1][0]
                return (7, [four_rank, kicker])
            if has_three and has_pair:
                three_rank = [r for r, c in rank_counts.items() if c == 3][0]
                pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
                return (6, [three_rank, pair_rank])
            if is_flush:
                return (5, sorted_ranks)
            if is_straight:
                return (4, [straight_high])
            if has_three:
                three_rank = [r for r, c in rank_counts.items() if c == 3][0]
                kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
                return (3, [three_rank] + kickers)
            if pair_count == 2:
                pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
                kicker = [r for r, c in rank_counts.items() if c == 1][0]
                return (2, pairs + [kicker])
            if has_pair:
                pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
                kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
                return (1, [pair_rank] + kickers)
            return (0, sorted_ranks)

        # ============================================================
        # 3. 比较两手牌
        # ============================================================
        def compare_hands(hand1, hand2, community_cards):
            cards1 = hand1 + community_cards
            cards2 = hand2 + community_cards
            score1 = evaluate_hand(cards1)
            score2 = evaluate_hand(cards2)
            if score1[0] > score2[0]:
                return 1
            elif score1[0] < score2[0]:
                return -1
            else:
                for a, b in zip(score1[1], score2[1]):
                    if a > b:
                        return 1
                    elif a < b:
                        return -1
                return 0

        # ============================================================
        # 4. 必胜牌检测 — 修复版：正确识别同花顺坚果
        # ============================================================
        def find_best_five_hand(cards):
            """从5张以上牌中找出最好的5张组合"""
            from itertools import combinations
            best_score = (0, [])
            for combo in combinations(cards, 5):
                score = evaluate_hand(list(combo))
                if score > best_score:
                    best_score = score
            return best_score

        def find_best_straight_flush_cards(hand, community):
            """
            找到最佳同花顺组合，返回 (最高牌rank, 同花花色, 完整的5张牌)
            如果没有同花顺则返回 None
            """
            all_cards = hand + community
            from itertools import combinations

            # 按花色分组
            suit_groups = {}
            for c in all_cards:
                s = c['suit']
                if s not in suit_groups:
                    suit_groups[s] = []
                suit_groups[s].append(c)

            best_sf = None  # (straight_high, suit, cards)

            for suit, group_cards in suit_groups.items():
                if len(group_cards) < 5:
                    continue
                # 取该花色牌按rank排序
                sranks = sorted([rank_values[c['rank']] for c in group_cards], reverse=True)

                # 检查是否有5连
                unique_sranks = sorted(set(sranks), reverse=True)
                for i in range(len(unique_sranks) - 4):
                    if unique_sranks[i] - unique_sranks[i + 4] == 4:
                        sf_high = unique_sranks[i]
                        # 找到这5张牌
                        sf_ranks = [unique_sranks[i] - 4, unique_sranks[i] - 3, unique_sranks[i] - 2,
                                    unique_sranks[i] - 1, unique_sranks[i]]
                        sf_cards = [c for c in group_cards if rank_values[c['rank']] in sf_ranks][:5]
                        if best_sf is None or sf_high > best_sf[0]:
                            best_sf = (sf_high, suit, sf_cards)
                        break
                # 检查 A-2-3-4-5 同花顺
                if set([14, 2, 3, 4, 5]).issubset(set(sranks)):
                    sf_ranks = [2, 3, 4, 5, 14]
                    sf_cards = [c for c in group_cards if rank_values[c['rank']] in sf_ranks][:5]
                    if best_sf is None or 5 > best_sf[0]:
                        best_sf = (5, suit, sf_cards)

            return best_sf

        def is_absolute_nuts(hand, community):
            """
            检测当前牌是否为坚果牌（绝对最大）
            返回: (是否必胜, 原因说明, 能赢你的牌列表)
            """
            full_cards = hand + community
            my_score = find_best_five_hand(full_cards)
            my_type = my_score[0]

            # 情况1: 皇家同花顺 → 100%必胜
            if my_type == 9:
                return (True, "皇家同花顺", [])

            # 情况2: 同花顺 → 关键修复：找最佳同花顺的最高牌
            if my_type == 8:
                best_sf = find_best_straight_flush_cards(hand, community)
                if best_sf is None:
                    return (False, "无法确定同花顺", [])

                sf_high, sf_suit, sf_cards = best_sf
                print(f"  🔍 最佳同花顺: {sf_high}高{sf_suit}, 牌: {[c['rank'] + c['suit'] for c in sf_cards]}")

                # 检查是否有更高的同花顺可能
                remaining_deck = []
                deck = generate_deck()
                used = set()
                for c in full_cards:
                    used.add(c['suit'] + c['rank'])
                remaining_deck = [c for c in deck if c['suit'] + c['rank'] not in used]

                beating_hands = []

                if sf_high < 14:  # 不是 A 高同花顺
                    # 检查是否存在 A 高的同花顺可能
                    for card in remaining_deck:
                        if card['suit'] == sf_suit and rank_values[card['rank']] == 14:
                            # 对手需要 A + 同花色的其他牌
                            for other in remaining_deck:
                                if other != card:
                                    beating_hands.append([card, other])

                if sf_high < 13:  # 不是 K 高同花顺
                    for card in remaining_deck:
                        if card['suit'] == sf_suit and rank_values[card['rank']] == 13:
                            for other in remaining_deck:
                                if other != card:
                                    beating_hands.append([card, other])

                if sf_high == 14:  # A 高同花顺 = 皇家同花顺 → 已在上一步处理
                    return (True, "皇家同花顺", [])

                if sf_high == 13:  # K 高同花顺
                    # 只有 A 高同花顺能击败它
                    has_A_in_suit = False
                    for c in full_cards:
                        if c['suit'] == sf_suit and rank_values[c['rank']] == 14:
                            has_A_in_suit = True
                            break
                    if not has_A_in_suit:
                        for card in remaining_deck:
                            if card['suit'] == sf_suit and rank_values[card['rank']] == 14:
                                for other in remaining_deck:
                                    if other != card:
                                        beating_hands.append([card, other])

                if not beating_hands:
                    return (True, f"同花顺({sf_high}高{sf_suit})，坚果牌！", [])

                return (False, f"同花顺({sf_high}高{sf_suit})，可能被打败", beating_hands)

            # 情况3: 四条 → 如果是四条A，基本锁定
            if my_type == 7:
                rank_counts = {}
                for c in full_cards:
                    r = rank_values[c['rank']]
                    rank_counts[r] = rank_counts.get(r, 0) + 1
                for r, count in rank_counts.items():
                    if count == 4 and r == 14:
                        return (True, "四条A，坚果牌！", [])
                max_four_rank = max([r for r, c in rank_counts.items() if c == 4], default=0)
                if max_four_rank < 14:
                    beating_hands = []
                    remaining_deck = []
                    deck = generate_deck()
                    used = set()
                    for c in full_cards:
                        used.add(c['suit'] + c['rank'])
                    remaining_deck = [c for c in deck if c['suit'] + c['rank'] not in used]
                    a_cards = [c for c in remaining_deck if c['rank'] == 'A']
                    for a in a_cards:
                        for other in remaining_deck:
                            if other != a:
                                beating_hands.append([a, other])
                    return (False, f"四条{max_four_rank}，对手可能有四条A", beating_hands)
                return (False, "非四条A", [])

            # 情况4: 葫芦 → 如果是A葫芦带K，基本锁定
            if my_type == 6:
                rank_counts = {}
                for c in full_cards:
                    r = rank_values[c['rank']]
                    rank_counts[r] = rank_counts.get(r, 0) + 1
                three_ranks = [r for r, c in rank_counts.items() if c >= 3]
                if three_ranks and max(three_ranks) == 14:
                    return (True, "A葫芦，坚果牌！", [])
                return (False, "非A葫芦", [])

            return (False, "非必胜牌型", [])

        # ============================================================
        # 5. 执行检测
        # ============================================================
        full_cards = hand + community
        my_score = find_best_five_hand(full_cards)
        my_type = my_score[0]

        # 牌型名称
        type_names = ['高牌', '一对', '两对', '三条', '顺子', '同花', '葫芦', '四条', '同花顺', '皇家同花顺']
        my_type_name = type_names[my_type] if my_type < len(type_names) else '未知'

        print(f"📋 当前牌型: {my_type_name}")

        # 检测是否为必胜牌
        is_nuts, reason, beating_hands = is_absolute_nuts(hand, community)

        if is_nuts:
            print(f"✅ 检测到必胜牌: {reason}")
            return jsonify({
                'success': True,
                'win': 10000,
                'lose': 0,
                'tie': 0,
                'total_hands': 10000,
                'opponent_range': '随机',
                'nuts': True,
                'nuts_reason': reason,
                'hand_type': my_type_name,
                'beating_hands': []
            })

        # ============================================================
        # 6. 列举能赢你的牌型 — 修复版：从except移到正常流程
        # ============================================================
        print(f"⚠️ 非必胜牌: {reason}")
        print(f"   能击败你的牌型: {len(beating_hands)} 种")

        # 将 beating_hands 转换为前端友好的格式
        beating_cards_display = []
        for bh in beating_hands[:20]:  # 最多显示20种
            beating_cards_display.append([
                {'suit': c['suit'], 'rank': c['rank']} for c in bh
            ])

        # ============================================================
        # 7. 执行蒙特卡洛模拟
        # ============================================================
        print("🔄 执行蒙特卡洛模拟...")

        # 生成牌堆
        deck = generate_deck()
        used_cards = set()
        for c in community + hand:
            used_cards.add(c['suit'] + c['rank'])
        deck = [c for c in deck if c['suit'] + c['rank'] not in used_cards]

        # 蒙特卡洛模拟
        num_simulations = min(5000, len(deck) // 2 * 500)
        community_needed = 5 - len(community)

        wins = 0
        losses = 0
        ties = 0

        for _ in range(num_simulations):
            shuffled = deck[:]
            random.shuffle(shuffled)

            full_community = community + shuffled[:community_needed]
            idx = community_needed

            opponent_hand = shuffled[idx:idx + 2]

            result = compare_hands(hand, opponent_hand, full_community)
            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                ties += 1

        print(f"✅ 胜率计算完成: 赢{wins} 输{losses} 平{ties}")

        return jsonify({
            'success': True,
            'win': wins,
            'lose': losses,
            'tie': ties,
            'total_hands': num_simulations,
            'opponent_range': '随机',
            'nuts': False,
            'nuts_reason': reason,
            'hand_type': my_type_name,
            'beating_hands': beating_cards_display
        })

    except Exception as e:
        print(f"❌ 胜率计算异常: {str(e)}")
        import traceback
        return jsonify({'success': False, 'message': str(e)}), 500


# ============================================================
# 路由4: 详细击败分析 — 枚举所有能击败当前牌型的对手手牌，按牌型分类
# ============================================================
@app.route('/analyze_beating', methods=['POST', 'OPTIONS'])
def analyze_beating_hands():
    """
    详细分析能击败当前牌型的所有对手手牌
    请求: { "community": [...], "hand": [...] }
    返回: {
        'my_type': '同花顺',
        'total_beating': 44,
        'total_possible': 990,
        'beating_probability': 4.4444,
        'beating_by_type': [
            {'type_rank': 9, 'type_name': '皇家同花顺', 'count': 44, 'hands': [[{suit,rank},...]], 'probability': 4.4444},
            ...
        ]
    }
    """
    if request.method == 'OPTIONS':
        return '', 200

    try:
        from itertools import combinations
        data = request.get_json()
        community = data.get('community', [])
        hand = data.get('hand', [])

        rank_values = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        suits_list = ['♠', '♥', '♦', '♣']
        type_names = ['高牌', '一对', '两对', '三条', '顺子', '同花', '葫芦', '四条', '同花顺', '皇家同花顺']

        def generate_deck():
            deck = []
            for suit in suits_list:
                for rank in rank_values.keys():
                    deck.append({'suit': suit, 'rank': rank})
            return deck

        def evaluate_hand(cards):
            if len(cards) < 5:
                return (0, [])
            ranks = [rank_values[c['rank']] for c in cards]
            suits = [c['suit'] for c in cards]
            rank_counts = {}
            for r in ranks:
                rank_counts[r] = rank_counts.get(r, 0) + 1
            sorted_ranks = sorted(ranks, reverse=True)
            is_flush = len(set(suits)) == 1
            unique_ranks = sorted(set(ranks), reverse=True)
            is_straight = False
            straight_high = 0
            if len(unique_ranks) >= 5:
                for i in range(len(unique_ranks) - 4):
                    if unique_ranks[i] - unique_ranks[i + 4] == 4:
                        is_straight = True
                        straight_high = unique_ranks[i]
                        break
                if set([14, 2, 3, 4, 5]).issubset(set(ranks)):
                    is_straight = True
                    straight_high = 5
            counts = list(rank_counts.values())
            pair_count = counts.count(2)
            if is_flush and is_straight:
                if straight_high == 14:
                    return (9, sorted_ranks)
                return (8, [straight_high] + sorted_ranks)
            if 4 in counts:
                four_rank = [r for r, c in rank_counts.items() if c == 4][0]
                kicker = [r for r, c in rank_counts.items() if c == 1][0]
                return (7, [four_rank, kicker])
            if 3 in counts and pair_count >= 1:
                three_rank = [r for r, c in rank_counts.items() if c == 3][0]
                pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
                return (6, [three_rank, pair_rank])
            if is_flush:
                return (5, sorted_ranks)
            if is_straight:
                return (4, [straight_high])
            if 3 in counts:
                three_rank = [r for r, c in rank_counts.items() if c == 3][0]
                kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
                return (3, [three_rank] + kickers)
            if pair_count == 2:
                pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
                kicker = [r for r, c in rank_counts.items() if c == 1][0]
                return (2, pairs + [kicker])
            if pair_count == 1:
                pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
                kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
                return (1, [pair_rank] + kickers)
            return (0, sorted_ranks)

        def find_best_five_hand(cards):
            from itertools import combinations
            best = (0, [])
            for combo in combinations(cards, 5):
                score = evaluate_hand(list(combo))
                if score > best:
                    best = score
            return best

        def compare_hands(h1, h2, comm):
            s1 = find_best_five_hand(h1 + comm)
            s2 = find_best_five_hand(h2 + comm)
            if s1 > s2:
                return 1
            elif s1 < s2:
                return -1
            else:
                return 0

        my_score = find_best_five_hand(hand + community)
        my_type_rank = my_score[0]
        my_type_name = type_names[my_type_rank] if my_type_rank < len(type_names) else '未知'

        deck = generate_deck()
        used = set()
        for c in community + hand:
            used.add(c['suit'] + c['rank'])
        remaining = [c for c in deck if c['suit'] + c['rank'] not in used]

        total_possible = 0
        beating_by_type = {}
        beating_count = 0

        for opp_hand in combinations(remaining, 2):
            opp_hand_list = list(opp_hand)
            total_possible += 1
            result = compare_hands(hand, opp_hand_list, community)
            if result < 0:
                opp_score = find_best_five_hand(opp_hand_list + community)
                opp_type = opp_score[0]
                opp_type_name = type_names[opp_type]
                beating_count += 1
                if opp_type_name not in beating_by_type:
                    beating_by_type[opp_type_name] = {'count': 0, 'hands': [], 'type_rank': opp_type}
                beating_by_type[opp_type_name]['count'] += 1
                if len(beating_by_type[opp_type_name]['hands']) < 10:
                    beating_by_type[opp_type_name]['hands'].append(opp_hand_list)

        sorted_beating = sorted(beating_by_type.values(), key=lambda x: x['type_rank'], reverse=True)

        result_list = []
        for bt in sorted_beating:
            result_list.append({
                'type_rank': bt['type_rank'],
                'type_name': type_names[bt['type_rank']],
                'count': bt['count'],
                'probability': round(bt['count'] / total_possible * 100, 4) if total_possible > 0 else 0,
                'hands': bt['hands']
            })

        return jsonify({
            'success': True,
            'my_type': my_type_name,
            'my_type_rank': my_type_rank,
            'total_beating': beating_count,
            'total_possible': total_possible,
            'beating_probability': round(beating_count / total_possible * 100, 4) if total_possible > 0 else 0,
            'beating_by_type': result_list
        })

    except Exception as e:
        print(f"❌ 击败分析异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


# ============================================================
# 起手牌排名页面
# ============================================================
@app.route('/rank')
def rank_page():
    return render_template('rank.html')


# ============================================================
# 启动服务
# ============================================================

# ============================================================
# 启动服务（云托管模式）
# ============================================================
port = int(os.environ.get('PORT', 8080))
print(f"🚀 Server running on port {port}")
app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
