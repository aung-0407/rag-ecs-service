#!/bin/bash

set -e

echo "========================================="
echo "Custom Open WebUI ECS Deployment"
echo "========================================="

# Load configuration
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    exit 1
fi

echo "Loading configuration from .env file..."
set -a
source <(sed -e 's/\r$//' -e '/^#/d' -e '/^$/d' .env)
set +a

# Check RAG_API_URL
if [ -z "$RAG_API_URL" ]; then
    echo "ERROR: RAG_API_URL is not set in .env file!"
    exit 1
fi

echo "RAG API URL: $RAG_API_URL"

# Service configuration
SERVICE_NAME="${WEBUI_SERVICE_NAME}"
CLUSTER_NAME="${RAG_CLUSTER_NAME}"
TASK_FAMILY="${WEBUI_TASK_FAMILY}"
ECR_REPO_NAME="open-webui-custom"

# Get AWS account ID
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "Auto-detecting AWS Account ID..."
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
fi
echo "AWS Account ID: $AWS_ACCOUNT_ID"

ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

echo ""
echo "Step 1: Create ECR repository for custom Open WebUI"
aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} --region ${AWS_REGION} 2>/dev/null || \
    aws ecr create-repository --repository-name ${ECR_REPO_NAME} --region ${AWS_REGION}

echo ""
echo "Step 2: Build and push custom Docker image"
echo "Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPO}

echo "Building custom Open WebUI image (this may take 5-10 minutes)..."
cd open-webui-custom
docker build -f Dockerfile.custom -t ${ECR_REPO_NAME}:latest .
cd ..

echo "Tagging image..."
docker tag ${ECR_REPO_NAME}:latest ${ECR_REPO}:latest

echo "Pushing to ECR..."
docker push ${ECR_REPO}:latest

echo ""
echo "Step 3: Ensure ECS cluster exists"
aws ecs describe-clusters --clusters ${CLUSTER_NAME} --region ${AWS_REGION} 2>/dev/null || \
    aws ecs create-cluster --cluster-name ${CLUSTER_NAME} --region ${AWS_REGION}

echo ""
echo "Step 4: Get existing IAM roles"
EXECUTION_ROLE_ARN=$(aws iam get-role --role-name ${ECS_TASK_EXECUTION_ROLE_NAME} --query 'Role.Arn' --output text)
echo "✓ Using execution role: ${EXECUTION_ROLE_ARN}"

echo ""
echo "Step 5: Get VPC PUBLIC subnets and security group"

SUBNETS=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=${VPC_ID}" "Name=tag:Name,Values=*public*" \
    --query 'Subnets[*].SubnetId' \
    --output text \
    --region ${AWS_REGION})

if [ -z "$SUBNETS" ] || [ "$SUBNETS" == "None" ]; then
    echo "No subnets with 'public' in name found, searching for subnets with IGW route..."
    IGW_ROUTE_TABLES=$(aws ec2 describe-route-tables \
        --filters "Name=vpc-id,Values=${VPC_ID}" \
        --query 'RouteTables[?Routes[?GatewayId!=`local` && starts_with(GatewayId, `igw-`)]].RouteTableId' \
        --output text \
        --region ${AWS_REGION})
    
    if [ -n "$IGW_ROUTE_TABLES" ]; then
        for RT_ID in $IGW_ROUTE_TABLES; do
            RT_SUBNETS=$(aws ec2 describe-route-tables \
                --route-table-ids ${RT_ID} \
                --query 'RouteTables[0].Associations[?SubnetId!=`null`].SubnetId' \
                --output text \
                --region ${AWS_REGION})
            SUBNETS="${SUBNETS} ${RT_SUBNETS}"
        done
        SUBNETS=$(echo $SUBNETS | xargs)
    fi
fi

if [ -z "$SUBNETS" ] || [ "$SUBNETS" == "None" ]; then
    echo "ERROR: No public subnets found in VPC ${VPC_ID}!"
    exit 1
fi

echo "Found public subnets: ${SUBNETS}"

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
echo "Step 6: Register ECS task definition"

cat > task-definition.json <<EOF
{
  "family": "${TASK_FAMILY}",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "${EXECUTION_ROLE_ARN}",
  "containerDefinitions": [
    {
      "name": "${SERVICE_NAME}",
      "image": "${ECR_REPO}:latest",
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
        },
        {
          "name": "WEBUI_NAME",
          "value": "Toshiba RAG System"
        },
        {
          "name": "DEFAULT_USER_ROLE",
          "value": "user"
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

SUBNET_IDS=$(echo $SUBNETS | tr ' ' ',')

SERVICE_EXISTS=$(aws ecs describe-services \
    --cluster ${CLUSTER_NAME} \
    --services ${SERVICE_NAME} \
    --region ${AWS_REGION} \
    --query 'services[0].status' \
    --output text 2>/dev/null)

if [ "$SERVICE_EXISTS" == "ACTIVE" ]; then
    echo "Updating existing service with custom image..."
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
echo "Custom Deployment Complete!"
echo "========================================="
echo ""
echo "Waiting for tasks to start (this may take 2-3 minutes)..."
sleep 60

TASK_ARN=$(aws ecs list-tasks \
    --cluster ${CLUSTER_NAME} \
    --service-name ${SERVICE_NAME} \
    --region ${AWS_REGION} \
    --query 'taskArns[0]' \
    --output text)

if [ -n "$TASK_ARN" ] && [ "$TASK_ARN" != "None" ]; then
    ENI_ID=$(aws ecs describe-tasks \
        --cluster ${CLUSTER_NAME} \
        --tasks ${TASK_ARN} \
        --region ${AWS_REGION} \
        --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
        --output text)
    
    if [ -n "$ENI_ID" ]; then
        PUBLIC_IP=$(aws ec2 describe-network-interfaces \
            --network-interface-ids ${ENI_ID} \
            --region ${AWS_REGION} \
            --query 'NetworkInterfaces[0].Association.PublicIp' \
            --output text)
        
        echo ""
        echo "========================================="
        echo "Custom Open WebUI is now running!"
        echo "========================================="
        echo ""
        echo "Public IP: ${PUBLIC_IP}"
        echo "Web Interface: http://${PUBLIC_IP}:8080"
        echo ""
        echo "Connected to RAG API: ${RAG_API_URL}"
        echo ""
    fi
fi

echo ""
echo "To view logs:"
echo "  aws logs tail /ecs/${SERVICE_NAME} --follow --region ${AWS_REGION}"
echo ""