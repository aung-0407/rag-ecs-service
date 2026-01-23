#!/bin/bash

# Open WebUI ECS Deployment Script (Using Existing Infrastructure)
# This script uses your existing VPC, security groups, and IAM roles

set -e  # Exit on error

echo "========================================="
echo "Open WebUI ECS Deployment"
echo "Using Your Existing Infrastructure"
echo "========================================="

# Load configuration from .env file
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    exit 1
fi

echo "Loading configuration from .env file..."
set -a
source <(sed -e 's/\r$//' -e '/^#/d' -e '/^$/d' .env)
set +a

# Check if RAG_API_URL is set
if [ -z "$RAG_API_URL" ]; then
    echo ""
    echo "ERROR: RAG_API_URL is not set in .env file!"
    echo "Please deploy RAG API first and update .env"
    echo ""
    exit 1
fi

echo "RAG API URL: $RAG_API_URL"

# Service configuration
SERVICE_NAME="${WEBUI_SERVICE_NAME}"
CLUSTER_NAME="${RAG_CLUSTER_NAME}"
TASK_FAMILY="${WEBUI_TASK_FAMILY}"

# Get AWS account ID
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "Auto-detecting AWS Account ID..."
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
fi
echo "AWS Account ID: $AWS_ACCOUNT_ID"

echo ""
echo "Step 1: Ensure ECS cluster exists"
aws ecs describe-clusters --clusters ${CLUSTER_NAME} --region ${AWS_REGION} 2>/dev/null || \
    aws ecs create-cluster --cluster-name ${CLUSTER_NAME} --region ${AWS_REGION}

echo ""
echo "Step 2: Get existing IAM roles"
EXECUTION_ROLE_ARN=$(aws iam get-role --role-name ${ECS_TASK_EXECUTION_ROLE_NAME} --query 'Role.Arn' --output text)
echo "✓ Using execution role: ${EXECUTION_ROLE_ARN}"

echo ""
echo "Step 3: Get VPC subnets and existing security group"

# Get subnets
SUBNETS=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=${VPC_ID}" \
    --query 'Subnets[*].SubnetId' \
    --output text \
    --region ${AWS_REGION})

echo "Found subnets: ${SUBNETS}"

# Get existing ECS security group
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=${ECS_SECURITY_GROUP_NAME}" "Name=vpc-id,Values=${VPC_ID}" \
    --query 'SecurityGroups[0].GroupId' \
    --output text \
    --region ${AWS_REGION})

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
    echo "ERROR: Security group '${ECS_SECURITY_GROUP_NAME}' not found!"
    exit 1
fi

echo "✓ Using existing security group: ${SG_ID} (${ECS_SECURITY_GROUP_NAME})"

# Check if port 8080 is allowed
PORT_8080_ALLOWED=$(aws ec2 describe-security-groups \
    --group-ids ${SG_ID} \
    --query "SecurityGroups[0].IpPermissions[?FromPort==\`8080\` && ToPort==\`8080\`] | length(@)" \
    --output text \
    --region ${AWS_REGION})

if [ "$PORT_8080_ALLOWED" == "0" ]; then
    echo "Adding inbound rule for port 8080..."
    aws ec2 authorize-security-group-ingress \
        --group-id ${SG_ID} \
        --protocol tcp \
        --port 8080 \
        --cidr 0.0.0.0/0 \
        --region ${AWS_REGION} || echo "Rule might already exist"
fi

echo ""
echo "Step 4: Register ECS task definition"

cat > task-definition.json <<EOF
{
  "family": "${TASK_FAMILY}",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "${CPU}",
  "memory": "${MEMORY}",
  "executionRoleArn": "${EXECUTION_ROLE_ARN}",
  "containerDefinitions": [
    {
      "name": "${SERVICE_NAME}",
      "image": "ghcr.io/open-webui/open-webui:main",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OPENAI_API_BASE_URL",
          "value": "${RAG_API_URL}/v1"
        },
        {
          "name": "OPENAI_API_KEY",
          "value": "dummy-key-not-needed"
        },
        {
          "name": "WEBUI_AUTH",
          "value": "true"
        },
        {
          "name": "ENABLE_SIGNUP",
          "value": "true"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/${SERVICE_NAME}",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "ecs",
          "awslogs-create-group": "true"
        }
      }
    }
  ]
}
EOF

aws ecs register-task-definition \
    --cli-input-json file://task-definition.json \
    --region ${AWS_REGION}

rm task-definition.json

echo ""
echo "Step 5: Create or update ECS service"

# Convert subnets to comma-separated list
SUBNET_IDS=$(echo $SUBNETS | tr ' ' ',')

# Check if service exists
SERVICE_EXISTS=$(aws ecs describe-services \
    --cluster ${CLUSTER_NAME} \
    --services ${SERVICE_NAME} \
    --region ${AWS_REGION} \
    --query 'services[0].status' \
    --output text 2>/dev/null)

if [ "$SERVICE_EXISTS" == "ACTIVE" ]; then
    echo "Updating existing service..."
    aws ecs update-service \
        --cluster ${CLUSTER_NAME} \
        --service ${SERVICE_NAME} \
        --task-definition ${TASK_FAMILY} \
        --force-new-deployment \
        --region ${AWS_REGION}
else
    echo "Creating new service..."
    aws ecs create-service \
        --cluster ${CLUSTER_NAME} \
        --service-name ${SERVICE_NAME} \
        --task-definition ${TASK_FAMILY} \
        --desired-count 1 \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[${SUBNET_IDS}],securityGroups=[${SG_ID}],assignPublicIp=ENABLED}" \
        --region ${AWS_REGION}
fi

echo ""
echo "========================================="
echo "Deployment Complete!"
echo "========================================="
echo ""
echo "Waiting for tasks to start (this may take 2-3 minutes)..."
sleep 60

# Get task ARN
TASK_ARN=$(aws ecs list-tasks \
    --cluster ${CLUSTER_NAME} \
    --service-name ${SERVICE_NAME} \
    --region ${AWS_REGION} \
    --query 'taskArns[0]' \
    --output text)

if [ -n "$TASK_ARN" ] && [ "$TASK_ARN" != "None" ]; then
    # Get ENI ID
    ENI_ID=$(aws ecs describe-tasks \
        --cluster ${CLUSTER_NAME} \
        --tasks ${TASK_ARN} \
        --region ${AWS_REGION} \
        --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
        --output text)
    
    if [ -n "$ENI_ID" ]; then
        # Get public IP
        PUBLIC_IP=$(aws ec2 describe-network-interfaces \
            --network-interface-ids ${ENI_ID} \
            --region ${AWS_REGION} \
            --query 'NetworkInterfaces[0].Association.PublicIp' \
            --output text)
        
        echo ""
        echo "========================================="
        echo "✅ Open WebUI is now running!"
        echo "========================================="
        echo ""
        echo "Public IP: ${PUBLIC_IP}"
        echo "Web Interface: http://${PUBLIC_IP}:8080"
        echo ""
        echo "Next steps:"
        echo "1. Open http://${PUBLIC_IP}:8080 in your browser"
        echo "2. Sign up for an account (first user becomes admin)"
        echo "3. Select 'rag-model' from the model dropdown"
        echo "4. Start chatting with your documents!"
        echo ""
        echo "Connected to RAG API: ${RAG_API_URL}"
        echo ""
    fi
fi

echo ""
echo "To view logs:"
echo "  aws logs tail /ecs/${SERVICE_NAME} --follow --region ${AWS_REGION}"
echo ""