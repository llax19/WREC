#!/usr/bin/env python3
"""生成用于服务实验的请求清单和多档 prompt 数据集。

这个脚本负责批量构造实验输入样本，输出格式与 `log_vllm_requests.py`
兼容，适合作为 `data/prompts/*.jsonl` 的来源。脚本会基于预设主题、
语言模板和输出长度档位，生成带有分组信息和目标生成长度的请求集合，
从而让后续实验在低、中、高负载下都能复用同一套输入规范。

典型用途：
1. 初始化 `debug_requests.jsonl`、`eval_requests.jsonl` 等实验输入文件。
2. 控制短/中/长输出样本比例，保证负载测试更稳定。
3. 在不手工写样本的前提下快速生成可重复的实验数据。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


GROUP_CONFIG = {
    "short": {"target_max_new_tokens": 64, "count": 96, "debug_count": 9},
    "medium": {"target_max_new_tokens": 128, "count": 96, "debug_count": 9},
    "long": {"target_max_new_tokens": 256, "count": 48, "debug_count": 6},
}


SUBJECTS = [
    {
        "id": "battery_recycling",
        "domain": "sustainability",
        "topic": {"zh": "社区电池回收服务", "en": "a community battery recycling service"},
        "context": {
            "zh": "该服务希望在商场和校园投放回收箱，但面临回收频率不稳定、居民参与度有限和危险品运输成本较高的问题。",
            "en": "The service plans to place collection boxes in malls and campuses, but pickup demand is uneven, user participation is limited, and hazardous transport is expensive.",
        },
        "tags": ["sustainability", "operations"],
    },
    {
        "id": "museum_guide",
        "domain": "culture",
        "topic": {"zh": "博物馆语音导览应用", "en": "a museum audio-guide application"},
        "context": {
            "zh": "用户希望快速找到适合儿童、游客和研究者三类人群的讲解内容，同时减少排队借设备的时间。",
            "en": "Visitors want guidance tailored to children, tourists, and researchers while reducing the wait for shared devices.",
        },
        "tags": ["culture", "personalization"],
    },
    {
        "id": "school_tutoring",
        "domain": "education",
        "topic": {"zh": "校内课后辅导平台", "en": "a school after-class tutoring platform"},
        "context": {
            "zh": "平台需要在考试周处理更多提问，并兼顾不同年级学生的学习差异和教师答疑负担。",
            "en": "The platform receives far more questions during exam weeks and must balance grade-level differences with teacher workload.",
        },
        "tags": ["education", "capacity"],
    },
    {
        "id": "grocery_delivery",
        "domain": "retail",
        "topic": {"zh": "同城生鲜配送系统", "en": "a local grocery delivery system"},
        "context": {
            "zh": "高峰期订单集中、冷链配送成本高，而且用户对延迟和缺货都非常敏感。",
            "en": "Orders spike at peak hours, cold-chain delivery is costly, and customers are sensitive to both delay and stockouts.",
        },
        "tags": ["retail", "logistics"],
    },
    {
        "id": "clinic_booking",
        "domain": "healthcare",
        "topic": {"zh": "社区门诊预约助手", "en": "a neighborhood clinic booking assistant"},
        "context": {
            "zh": "用户需要区分普通问诊和紧急情况，系统还要处理医生排班变动和重复预约问题。",
            "en": "Users need help distinguishing routine visits from urgent cases, while the system must handle doctor schedule changes and duplicate bookings.",
        },
        "tags": ["healthcare", "scheduling"],
    },
    {
        "id": "travel_planner",
        "domain": "travel",
        "topic": {"zh": "多城市旅行规划工具", "en": "a multi-city travel planning tool"},
        "context": {
            "zh": "用户希望在预算有限的前提下兼顾交通时间、景点开放时间和同行人的偏好差异。",
            "en": "Travelers want to balance a fixed budget with transit time, opening hours, and differing group preferences.",
        },
        "tags": ["travel", "planning"],
    },
    {
        "id": "budget_app",
        "domain": "finance",
        "topic": {"zh": "个人预算管理应用", "en": "a personal budgeting application"},
        "context": {
            "zh": "用户希望快速识别支出异常，但又不希望系统频繁发出误报提醒。",
            "en": "Users want quick detection of unusual spending without receiving too many false alarms.",
        },
        "tags": ["finance", "risk"],
    },
    {
        "id": "cloud_alerting",
        "domain": "software",
        "topic": {"zh": "云成本告警面板", "en": "a cloud cost alerting dashboard"},
        "context": {
            "zh": "团队经常在月底才发现预算超支，希望更早识别成本飙升的服务和原因。",
            "en": "Teams often notice overspending only at month end and want earlier signals about which services are driving the increase.",
        },
        "tags": ["software", "cost"],
    },
    {
        "id": "farm_irrigation",
        "domain": "agriculture",
        "topic": {"zh": "农田灌溉监测网络", "en": "a farm irrigation monitoring network"},
        "context": {
            "zh": "不同地块的土壤湿度变化不一致，传感器数据还会受到天气和电量不足影响。",
            "en": "Soil moisture shifts differently across plots, and sensor quality can degrade under weather changes and low battery.",
        },
        "tags": ["agriculture", "sensors"],
    },
    {
        "id": "phishing_training",
        "domain": "security",
        "topic": {"zh": "企业钓鱼邮件培训项目", "en": "a corporate phishing-awareness training program"},
        "context": {
            "zh": "员工对重复培训容易失去耐心，但安全团队仍需要持续评估风险变化和薄弱群体。",
            "en": "Employees become fatigued by repeated training, yet the security team still needs ongoing visibility into changing risk and vulnerable groups.",
        },
        "tags": ["security", "training"],
    },
    {
        "id": "warehouse_picker",
        "domain": "logistics",
        "topic": {"zh": "仓库拣货路线优化工具", "en": "a warehouse picking-route optimizer"},
        "context": {
            "zh": "订单结构波动较大，热门商品区域经常拥堵，导致新员工和熟练员工效率差距明显。",
            "en": "Order mix changes quickly, popular aisles become congested, and the productivity gap between new and experienced workers is large.",
        },
        "tags": ["logistics", "optimization"],
    },
    {
        "id": "subtitle_editor",
        "domain": "media",
        "topic": {"zh": "视频字幕编辑平台", "en": "a video subtitle editing platform"},
        "context": {
            "zh": "创作者需要同时处理翻译、时间轴对齐和术语统一，交付周期通常很紧。",
            "en": "Creators need to handle translation, timing alignment, and terminology consistency under tight deadlines.",
        },
        "tags": ["media", "editing"],
    },
    {
        "id": "transit_info",
        "domain": "public_service",
        "topic": {"zh": "城市公共交通信息助手", "en": "a city public transit information assistant"},
        "context": {
            "zh": "乘客最关心绕行通知和换乘建议，但高峰期线路调整频繁且信息来源不统一。",
            "en": "Passengers care most about detours and transfer advice, but route changes are frequent at rush hour and data sources are inconsistent.",
        },
        "tags": ["public_service", "mobility"],
    },
    {
        "id": "smart_home_energy",
        "domain": "iot",
        "topic": {"zh": "家庭能源管理面板", "en": "a home energy management dashboard"},
        "context": {
            "zh": "系统需要帮助住户理解哪些家电在高峰时段最耗电，同时避免让建议看起来过于复杂。",
            "en": "The system should help residents understand which appliances drive peak-time usage without making the advice feel overly technical.",
        },
        "tags": ["iot", "energy"],
    },
    {
        "id": "bookstore_search",
        "domain": "search",
        "topic": {"zh": "独立书店检索与推荐页面", "en": "an independent bookstore search and recommendation page"},
        "context": {
            "zh": "顾客常常只记得模糊主题或片段描述，店员希望减少反复沟通的时间。",
            "en": "Customers often remember only vague themes or fragments, and staff want to reduce the back-and-forth required to help them.",
        },
        "tags": ["search", "recommendation"],
    },
    {
        "id": "course_forum",
        "domain": "community",
        "topic": {"zh": "在线课程讨论区助手", "en": "an online course forum assistant"},
        "context": {
            "zh": "讨论区在作业截止前会突然涌入大量重复问题，助教希望更快发现高频困惑点。",
            "en": "Right before deadlines the forum is flooded with repeated questions, and teaching assistants want quicker visibility into common pain points.",
        },
        "tags": ["community", "education"],
    },
    {
        "id": "restaurant_menu",
        "domain": "hospitality",
        "topic": {"zh": "餐厅菜单数字化服务", "en": "a restaurant menu digitization service"},
        "context": {
            "zh": "门店同时面向游客和本地顾客，既要突出招牌菜，又要处理过敏原和库存变化说明。",
            "en": "The service targets both tourists and locals, and it must highlight signature dishes while explaining allergens and changing availability.",
        },
        "tags": ["hospitality", "content"],
    },
    {
        "id": "recruiting_portal",
        "domain": "hr",
        "topic": {"zh": "校园招聘信息门户", "en": "a campus recruiting information portal"},
        "context": {
            "zh": "学生最关心岗位门槛和投递截止时间，企业则希望减少不匹配申请的数量。",
            "en": "Students care most about role requirements and deadlines, while employers want fewer clearly mismatched applications.",
        },
        "tags": ["hr", "matching"],
    },
]


TASK_FAMILIES = [
    {
        "id": "explanation",
        "tags": ["explanation", "qa"],
        "templates": {
            "short": {
                "zh": "请用三到四句话解释{topic}。背景：{context}",
                "en": "Explain {topic} in three or four sentences. Background: {context}",
            },
            "medium": {
                "zh": "请从产品或服务设计角度解释{topic}为什么值得关注，并结合以下背景给出一个简单例子：{context}",
                "en": "Explain why {topic} deserves attention from a product or service perspective, and use this background to include one simple example: {context}",
            },
            "long": {
                "zh": "请围绕{topic}写一段较长分析，结合以下背景说明问题成因、关键约束和改进方向：{context}",
                "en": "Write a detailed analysis of {topic}, using this background to discuss root causes, key constraints, and possible improvements: {context}",
            },
        },
    },
    {
        "id": "summarization",
        "tags": ["summarization", "compression"],
        "templates": {
            "short": {
                "zh": "请将下面这段信息压缩成简洁摘要，不超过三句话：{context}",
                "en": "Compress the following information into a concise summary in no more than three sentences: {context}",
            },
            "medium": {
                "zh": "请基于以下背景写一段结构化摘要，覆盖目标、困难和潜在影响：{context}",
                "en": "Write a structured summary of the following background, covering goals, difficulties, and likely impact: {context}",
            },
            "long": {
                "zh": "请根据以下背景写一份较完整摘要，分为现状、问题和建议三部分：{context}",
                "en": "Produce a more complete summary from the following background, organized into current state, problems, and recommendations: {context}",
            },
        },
    },
    {
        "id": "rewriting",
        "tags": ["rewriting", "style"],
        "templates": {
            "short": {
                "zh": "请把关于{topic}的说明改写得更清晰易懂。原始背景：{context}",
                "en": "Rewrite the description of {topic} so it is clearer and easier to understand. Original background: {context}",
            },
            "medium": {
                "zh": "请把下面这段与{topic}相关的说明改写成适合普通用户阅读的版本：{context}",
                "en": "Rewrite the following note about {topic} into a version suitable for a general audience: {context}",
            },
            "long": {
                "zh": "请将与{topic}相关的以下材料改写成一段正式但易读的长说明，适合放入产品文档：{context}",
                "en": "Rewrite the following material about {topic} into a longer formal but readable description suitable for product documentation: {context}",
            },
        },
    },
    {
        "id": "extraction",
        "tags": ["extraction", "structured-output"],
        "templates": {
            "short": {
                "zh": "请从以下信息中提取三个最关键的点：{context}",
                "en": "Extract the three most important points from the following information: {context}",
            },
            "medium": {
                "zh": "请根据以下背景提取关键信息，并按“目标、风险、限制”三个小标题组织：{context}",
                "en": "Extract key information from the following background and organize it under the headings Goal, Risk, and Constraint: {context}",
            },
            "long": {
                "zh": "请阅读以下背景，提取关键实体、主要问题和后续动作建议，并用较完整段落呈现：{context}",
                "en": "Read the following background and extract the main entities, major problems, and suggested next actions in a fuller paragraph form: {context}",
            },
        },
    },
    {
        "id": "comparison",
        "tags": ["comparison", "decision-making"],
        "templates": {
            "short": {
                "zh": "请简要比较在{topic}中采用“自动化优先”和“人工复核优先”两种做法的差异。背景：{context}",
                "en": "Briefly compare an automation-first approach with a human-review-first approach for {topic}. Background: {context}",
            },
            "medium": {
                "zh": "请比较在{topic}场景下追求更低成本与追求更高质量两种策略的优缺点。背景：{context}",
                "en": "Compare the pros and cons of a lower-cost strategy versus a higher-quality strategy for {topic}. Background: {context}",
            },
            "long": {
                "zh": "请围绕{topic}详细比较两条可行路线：快速上线方案与稳健迭代方案，并结合以下背景分析适用条件：{context}",
                "en": "For {topic}, compare a fast-launch path with a more conservative iterative path, and use this background to analyze when each is appropriate: {context}",
            },
        },
    },
    {
        "id": "planning",
        "tags": ["planning", "operations"],
        "templates": {
            "short": {
                "zh": "请为{topic}给出一个简短三步计划。背景：{context}",
                "en": "Provide a short three-step plan for {topic}. Background: {context}",
            },
            "medium": {
                "zh": "请为{topic}制定一个中等详细度的执行计划，覆盖优先级、资源和风险。背景：{context}",
                "en": "Create a moderately detailed execution plan for {topic}, covering priorities, resources, and risks. Background: {context}",
            },
            "long": {
                "zh": "请为{topic}写一份较完整的实施方案，至少包含阶段划分、关键指标和应急预案。背景：{context}",
                "en": "Write a fuller implementation plan for {topic}, including phases, key metrics, and a fallback plan. Background: {context}",
            },
        },
    },
    {
        "id": "troubleshooting",
        "tags": ["troubleshooting", "diagnosis"],
        "templates": {
            "short": {
                "zh": "如果{topic}出现用户投诉增加的情况，请用几句话判断最可能的原因。背景：{context}",
                "en": "If {topic} starts receiving more user complaints, identify the most likely causes in a few sentences. Background: {context}",
            },
            "medium": {
                "zh": "请分析{topic}近期表现变差的可能原因，并给出一个排查顺序。背景：{context}",
                "en": "Analyze why {topic} may have degraded recently and suggest an order for investigation. Background: {context}",
            },
            "long": {
                "zh": "请以排障备忘录的口吻分析{topic}出现持续问题的可能来源、证据和修复优先级。背景：{context}",
                "en": "Write a troubleshooting memo for {topic}, analyzing likely sources of the persistent issue, supporting evidence, and repair priorities. Background: {context}",
            },
        },
    },
    {
        "id": "argumentation",
        "tags": ["argumentation", "tradeoff"],
        "templates": {
            "short": {
                "zh": "请就{topic}给出一个简短立场，说明为什么现在值得投入改进。背景：{context}",
                "en": "Take a brief position on {topic} and explain why it is worth improving now. Background: {context}",
            },
            "medium": {
                "zh": "请围绕{topic}写一段论证，说明为什么管理者应当优先处理这个问题。背景：{context}",
                "en": "Write an argument for why managers should prioritize work on {topic}. Background: {context}",
            },
            "long": {
                "zh": "请围绕{topic}写一段较长论证，讨论投入收益、潜在反对意见以及折中方案。背景：{context}",
                "en": "Develop a longer argument about {topic}, discussing return on effort, likely objections, and a practical compromise. Background: {context}",
            },
        },
    },
]


REQUIREMENTS = {
    "short": {
        "zh": [
            "请保持结论直接。",
            "不要列出超过三点。",
            "如果合适，请提到一个具体指标。",
        ],
        "en": [
            "Keep the conclusion direct.",
            "Do not list more than three points.",
            "Mention one concrete metric if it helps.",
        ],
    },
    "medium": {
        "zh": [
            "请让结构清晰，适合实验记录。",
            "尽量同时说明收益和风险。",
            "如有必要，可以分成两个自然段。",
        ],
        "en": [
            "Keep the structure clear enough for an experiment note.",
            "Try to mention both benefits and risks.",
            "You may use two short paragraphs if helpful.",
        ],
    },
    "long": {
        "zh": [
            "请加入简短小标题。",
            "结尾请给出一句总结。",
            "尽量兼顾业务目标与实施难点。",
        ],
        "en": [
            "Include short section headers.",
            "End with a one-sentence takeaway.",
            "Balance business goals with implementation difficulty.",
        ],
    },
}


def build_prompt(group: str, lang: str, task_family: dict, subject: dict, variant_idx: int) -> str:
    base = task_family["templates"][group][lang].format(
        topic=subject["topic"][lang],
        context=subject["context"][lang],
    )
    requirement = REQUIREMENTS[group][lang][variant_idx % len(REQUIREMENTS[group][lang])]
    if lang == "zh":
        return f"{base}{requirement}"
    return f"{base} {requirement}"


def candidate_rows(group: str, target_max_new_tokens: int, rng: random.Random) -> list[dict]:
    candidates: list[dict] = []
    for task_family in TASK_FAMILIES:
        for lang in ("zh", "en"):
            variant_count = len(REQUIREMENTS[group][lang])
            for subject in SUBJECTS:
                for variant_idx in range(variant_count):
                    candidates.append(
                        {
                            "group": group,
                            "topic": subject["id"],
                            "domain": subject["domain"],
                            "task_family": task_family["id"],
                            "tags": sorted(set(subject["tags"] + task_family["tags"])),
                            "lang": lang,
                            "prompt": build_prompt(group, lang, task_family, subject, variant_idx),
                            "target_max_new_tokens": target_max_new_tokens,
                            "source": "synthetic_diverse_workload_v2",
                        }
                    )
    rng.shuffle(candidates)
    return candidates


def select_rows(candidates: list[dict], desired_count: int) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = {}
    for row in candidates:
        buckets.setdefault((row["task_family"], row["lang"]), []).append(row)

    ordered_keys = sorted(buckets)
    selected: list[dict] = []
    while len(selected) < desired_count:
        progress = False
        for key in ordered_keys:
            bucket = buckets[key]
            if bucket and len(selected) < desired_count:
                selected.append(bucket.pop())
                progress = True
        if not progress:
            break
    return selected


def generate_rows(count_by_group: dict[str, int], prefix: str) -> list[dict]:
    rng = random.Random(42)
    rows: list[dict] = []
    next_id = 1

    for group in ("short", "medium", "long"):
        candidates = candidate_rows(group, GROUP_CONFIG[group]["target_max_new_tokens"], rng)
        selected = select_rows(candidates, count_by_group[group])
        for row in selected:
            row = dict(row)
            row["id"] = f"{prefix}-{next_id:04d}"
            next_id += 1
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/workspace/data/prompts"),
        help="Directory used to save generated JSONL manifests.",
    )
    args = parser.parse_args()

    debug_rows = generate_rows(
        {group: config["debug_count"] for group, config in GROUP_CONFIG.items()},
        prefix="dbg",
    )
    eval_rows = generate_rows(
        {group: config["count"] for group, config in GROUP_CONFIG.items()},
        prefix="eval",
    )

    write_jsonl(args.output_dir / "debug_requests.jsonl", debug_rows)
    write_jsonl(args.output_dir / "eval_requests.jsonl", eval_rows)
    write_jsonl(args.output_dir / "requests_template.jsonl", debug_rows)

    summary = {
        "debug_requests": len(debug_rows),
        "eval_requests": len(eval_rows),
        "group_counts": {
            "debug": {group: config["debug_count"] for group, config in GROUP_CONFIG.items()},
            "eval": {group: config["count"] for group, config in GROUP_CONFIG.items()},
        },
        "task_families": [task["id"] for task in TASK_FAMILIES],
        "domains": sorted({subject["domain"] for subject in SUBJECTS}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
