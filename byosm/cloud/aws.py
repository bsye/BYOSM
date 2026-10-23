from __future__ import annotations

from typing import Any

from ..rt import HEALTH_PATH, TOKEN_ENV

PORT = 8800
SCALE_CPU_PERCENT = 60


def _ref(name: str) -> dict[str, str]:
    return {"Ref": name}


def _parameters() -> dict[str, Any]:
    return {
        "ImageUri": {
            "Type": "String",
            "Description": "the pushed image, e.g. <account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>",
        },
        "VpcId": {"Type": "AWS::EC2::VPC::Id"},
        "SubnetIds": {
            "Type": "List<AWS::EC2::Subnet::Id>",
            "Description": "at least two public subnets for the load balancer",
        },
        "TokenSecretArn": {"Type": "String", "Description": f"Secrets Manager secret holding {TOKEN_ENV}"},
        "MinTasks": {"Type": "Number", "Default": 1},
        "MaxTasks": {"Type": "Number", "Default": 3},
    }


def _task(name: str, sizing: dict[str, Any]) -> dict[str, Any]:
    container = {
        "Name": name,
        "Image": _ref("ImageUri"),
        "PortMappings": [{"ContainerPort": PORT}],
        "ReadonlyRootFilesystem": True,
        "Secrets": [{"Name": TOKEN_ENV, "ValueFrom": _ref("TokenSecretArn")}],
        "LogConfiguration": {
            "LogDriver": "awslogs",
            "Options": {
                "awslogs-group": _ref("Logs"),
                "awslogs-region": _ref("AWS::Region"),
                "awslogs-stream-prefix": name,
            },
        },
    }
    return {
        "Type": "AWS::ECS::TaskDefinition",
        "Properties": {
            "Family": name,
            "RequiresCompatibilities": ["FARGATE"],
            "NetworkMode": "awsvpc",
            "Cpu": str(sizing["cpu"]),
            "Memory": str(sizing["memory"]),
            "ExecutionRoleArn": {"Fn::GetAtt": ["ExecutionRole", "Arn"]},
            "ContainerDefinitions": [container],
        },
    }


def _role() -> dict[str, Any]:
    trust = {"Effect": "Allow", "Principal": {"Service": "ecs-tasks.amazonaws.com"}, "Action": "sts:AssumeRole"}
    read_secret = {"Effect": "Allow", "Action": "secretsmanager:GetSecretValue", "Resource": _ref("TokenSecretArn")}
    return {
        "Type": "AWS::IAM::Role",
        "Properties": {
            "AssumeRolePolicyDocument": {"Version": "2012-10-17", "Statement": [trust]},
            "ManagedPolicyArns": ["arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"],
            "Policies": [
                {"PolicyName": "read-token", "PolicyDocument": {"Version": "2012-10-17", "Statement": [read_secret]}}
            ],
        },
    }


def _network() -> dict[str, Any]:
    def group(desc: str, ingress: dict[str, Any]) -> dict[str, Any]:
        return {
            "Type": "AWS::EC2::SecurityGroup",
            "Properties": {"GroupDescription": desc, "VpcId": _ref("VpcId"), "SecurityGroupIngress": [ingress]},
        }

    web = {"IpProtocol": "tcp", "FromPort": 80, "ToPort": 80, "CidrIp": "0.0.0.0/0"}
    lb = {"IpProtocol": "tcp", "FromPort": PORT, "ToPort": PORT, "SourceSecurityGroupId": _ref("BalancerGroup")}
    return {
        "BalancerGroup": group("HTTP to the load balancer", web),
        "TaskGroup": group("only the load balancer reaches the model", lb),
        "Balancer": {
            "Type": "AWS::ElasticLoadBalancingV2::LoadBalancer",
            "Properties": {"Subnets": _ref("SubnetIds"), "SecurityGroups": [_ref("BalancerGroup")]},
        },
        "Targets": {
            "Type": "AWS::ElasticLoadBalancingV2::TargetGroup",
            "Properties": {
                "VpcId": _ref("VpcId"),
                "Port": PORT,
                "Protocol": "HTTP",
                "TargetType": "ip",
                "HealthCheckPath": HEALTH_PATH,
            },
        },
        "Listener": {
            "Type": "AWS::ElasticLoadBalancingV2::Listener",
            "Properties": {
                "LoadBalancerArn": _ref("Balancer"),
                "Port": 80,
                "Protocol": "HTTP",
                "DefaultActions": [{"Type": "forward", "TargetGroupArn": _ref("Targets")}],
            },
        },
    }


def _service(name: str) -> dict[str, Any]:
    network = {
        "AwsvpcConfiguration": {
            "Subnets": _ref("SubnetIds"),
            "SecurityGroups": [_ref("TaskGroup")],
            "AssignPublicIp": "ENABLED",
        }
    }
    target = {"Fn::Sub": "service/${Cluster}/${Service.Name}"}
    return {
        "Service": {
            "Type": "AWS::ECS::Service",
            "DependsOn": "Listener",
            "Properties": {
                "Cluster": _ref("Cluster"),
                "LaunchType": "FARGATE",
                "DesiredCount": _ref("MinTasks"),
                "TaskDefinition": _ref("Task"),
                "NetworkConfiguration": network,
                "LoadBalancers": [{"ContainerName": name, "ContainerPort": PORT, "TargetGroupArn": _ref("Targets")}],
            },
        },
        "Scaling": {
            "Type": "AWS::ApplicationAutoScaling::ScalableTarget",
            "Properties": {
                "ServiceNamespace": "ecs",
                "ScalableDimension": "ecs:service:DesiredCount",
                "ResourceId": target,
                "MinCapacity": _ref("MinTasks"),
                "MaxCapacity": _ref("MaxTasks"),
            },
        },
        "ScaleOnCpu": {
            "Type": "AWS::ApplicationAutoScaling::ScalingPolicy",
            "Properties": {
                "PolicyName": f"{name}-cpu",
                "PolicyType": "TargetTrackingScaling",
                "ScalingTargetId": _ref("Scaling"),
                "TargetTrackingScalingPolicyConfiguration": {
                    "TargetValue": SCALE_CPU_PERCENT,
                    "PredefinedMetricSpecification": {"PredefinedMetricType": "ECSServiceAverageCPUUtilization"},
                },
            },
        },
    }


def template(name: str, sizing: dict[str, Any]) -> dict[str, Any]:
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": f"BYOSM {name} on ECS Fargate: {sizing['cpu']} CPU units, {sizing['memory']} MB ({sizing['why']})",
        "Parameters": _parameters(),
        "Resources": {
            "Cluster": {"Type": "AWS::ECS::Cluster"},
            "Logs": {"Type": "AWS::Logs::LogGroup", "Properties": {"RetentionInDays": 14}},
            "ExecutionRole": _role(),
            "Task": _task(name, sizing),
            **_network(),
            **_service(name),
        },
        "Outputs": {"Url": {"Value": {"Fn::Sub": "http://${Balancer.DNSName}"}}},
    }
