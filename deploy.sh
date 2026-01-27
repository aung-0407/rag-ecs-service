#!/bin/bash

# RAG API ECS Deployment Script (Using Existing Infrastructure)
# This script uses your existing VPC, security groups, and IAM roles

set -e  # Exit on error

echo "========================================="
echo "RAG API ECS Deployment"
echo "Using Your Existing Infrastructure"
echo "========================================="

# Load configuration from .env file
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please create a .env file with your configuration."
    exit 1
fi

echo "Loading configuration from .env file..."
set -a
source <(sed -e 's/\r$//' -e '/^#/d' -e '/^$/d' .env)
set +a

# Service configuration
SERVICE_NAME="${RAG_SERVICE_NAME}"
CLUSTER_NAME="${RAG_CLUSTER_NAME}"
ECR_REPO_NAME="${RAG_ECR_REPO_NAME}"
TASK_FAMILY="${RAG_TASK_FAMILY}"

# Get AWS account ID
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "Auto-detecting AWS Account ID..."
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
fi
echo "AWS Account ID: $AWS_ACCOUNT_ID"

# ECR repository URL
ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

echo ""
echo "Step 1: Create ECR repository (if not exists)"
aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} --region ${AWS_REGION} 2>/dev/null || \
    aws ecr create-repository --repository-name ${ECR_REPO_NAME} --region ${AWS_REGION}

echo ""
echo "Step 2: Build and push Docker image"
echo "Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPO}

echo "Building Docker image..."
docker build -t ${ECR_REPO_NAME}:latest .

echo "Tagging image..."
docker tag ${ECR_REPO_NAME}:latest ${ECR_REPO}:latest

echo "Pushing to ECR..."
docker push ${ECR_REPO}:latest

echo ""
echo "Step 3: Create ECS cluster (if not exists)"
aws ecs describe-clusters --clusters ${CLUSTER_NAME} --region ${AWS_REGION} 2>/dev/null || \
    aws ecs create-cluster --cluster-name ${CLUSTER_NAME} --region ${AWS_REGION}

echo ""
echo "Step 4: Get existing IAM roles"
echo "Using your existing IAM roles..."

# Get execution role ARN
EXECUTION_ROLE_ARN=$(aws iam get-role --role-name ${ECS_TASK_EXECUTION_ROLE_NAME} --query 'Role.Arn' --output text)
echo "✓ Execution Role: ${EXECUTION_ROLE_ARN}"

# Get task role ARN
TASK_ROLE_ARN=$(aws iam get-role --role-name ${ECS_TASK_ROLE_NAME} --query 'Role.Arn' --output text)
echo "✓ Task Role: ${TASK_ROLE_ARN}"

echo ""
echo "Step 5: Get VPC subnets and existing security group"

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
    echo "Please create it first or check the name in .env file"
    exit 1
fi

echo "✓ Using existing security group: ${SG_ID} (${ECS_SECURITY_GROUP_NAME})"

# Check if port 8000 is allowed
echo "Checking security group rules..."
PORT_8000_ALLOWED=$(aws ec2 describe-security-groups \
    --group-ids ${SG_ID} \
    --query "SecurityGroups[0].IpPermissions[?FromPort==\`8000\` && ToPort==\`8000\`] | length(@)" \
    --output text \
    --region ${AWS_REGION})

if [ "$PORT_8000_ALLOWED" == "0" ]; then
    echo "Adding inbound rule for port 8000..."
    aws ec2 authorize-security-group-ingress \
        --group-id ${SG_ID} \
        --protocol tcp \
        --port 8000 \
        --cidr 0.0.0.0/0 \
        --region ${AWS_REGION} || echo "Rule might already exist"
fi

echo ""
echo "Step 6: Register ECS task definition"

cat > task-definition.json <<EOF
{
  "family": "${TASK_FAMILY}",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "${CPU}",
  "memory": "${MEMORY}",
  "executionRoleArn": "${EXECUTION_ROLE_ARN}",
  "taskRoleArn": "${TASK_ROLE_ARN}",
  "containerDefinitions": [
    {
      "name": "${SERVICE_NAME}",
      "image": "${ECR_REPO}:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OPENSEARCH_ENDPOINT",
          "value": "${OPENSEARCH_ENDPOINT}"
        },
        {
          "name": "OPENSEARCH_INDEX",
          "value": "${OPENSEARCH_INDEX}"
        },
        {
          "name": "AWS_REGION",
          "value": "${AWS_REGION}"
        },
        {
          "name": "LLM_MODEL",
          "value": "${LLM_MODEL}"
        },
        {
          "name": "SECRET_NAME",
          "value": "${SECRET_NAME}"
        },
        {
          "name": "SECRET_NAME_ANTHROPIC",
          "value": "${SECRET_NAME_ANTHROPIC}"
        },
        {
          "name": "LLM_MODEL_ANTHROPIC",
          "value": "${LLM_MODEL_ANTHROPIC}"
        },
        {
          "name": "SECRET_NAME_GOOGLE",
          "value": "${SECRET_NAME_GOOGLE}"
        },
        {
          "name": "LLM_MODEL_GOOGLE",
          "value": "${LLM_MODEL_GOOGLE}"
        }
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
echo "Step 7: Create or update ECS service"

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
echo "Service: ${SERVICE_NAME}"
echo "Cluster: ${CLUSTER_NAME}"
echo "Region: ${AWS_REGION}"
echo "Security Group: ${SG_ID} (${ECS_SECURITY_GROUP_NAME})"
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
        echo "✅ RAG API is now running!"
        echo "========================================="
        echo ""
        echo "Public IP: ${PUBLIC_IP}"
        echo "API Endpoint: http://${PUBLIC_IP}:8000"
        echo ""
        echo "Test commands:"
        echo "  Health check: curl http://${PUBLIC_IP}:8000/health"
        echo "  List models: curl http://${PUBLIC_IP}:8000/v1/models"
        echo ""
        echo "========================================="
        echo "⚠️  IMPORTANT: Update your .env file!"
        echo "========================================="
        echo ""
        echo "Run this command to update .env:"
        echo "  sed -i 's|RAG_API_URL=.*|RAG_API_URL=http://${PUBLIC_IP}:8000|' .env"
        echo ""
        echo "Or manually edit .env and add:"
        echo "  RAG_API_URL=http://${PUBLIC_IP}:8000"
        echo ""
    fi
fi

echo ""
echo "To view logs:"
echo "  aws logs tail /ecs/${SERVICE_NAME} --follow --region ${AWS_REGION}"
echo ""
echo "To check service status:"
echo "  aws ecs describe-services --cluster ${CLUSTER_NAME} --services ${SERVICE_NAME} --region ${AWS_REGION}"
echo ""
echo "Next step: Deploy Open WebUI with ./deploy-open-webui.sh"
echo ""