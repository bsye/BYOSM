from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .aws import template

STEPS = """# Deploy {name} to AWS

Sized for {rps:g} requests per second: {cpu} CPU units and {memory} MB ({why}).
Nothing here holds a credential: the AWS CLI uses your own sign-in (`aws sso login` or a profile).

```bash
aws ecr create-repository --repository-name byosm/{name}
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
byosm containerize {name} --tag <account>.dkr.ecr.<region>.amazonaws.com/byosm/{name}:{tag}
docker push <account>.dkr.ecr.<region>.amazonaws.com/byosm/{name}:{tag}
aws secretsmanager create-secret --name byosm/{name}/token --secret-string "$(openssl rand -base64 32)"
aws cloudformation deploy --stack-name byosm-{name} --template-file template.yaml --capabilities CAPABILITY_IAM \\
  --parameter-overrides ImageUri=<image above> VpcId=<vpc> SubnetIds=<subnet-a>,<subnet-b> TokenSecretArn=<secret arn>
```

Call it with `curl -H "Authorization: Bearer <token>" -d '{{"input": "..."}}' <Url>/v1/{name}/predict`.
The load balancer listens on HTTP; put an HTTPS listener with your certificate in front before sending real data.
"""


def write(out: Path, name: str, sizing: dict[str, Any], tag: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "template.yaml").write_text(yaml.safe_dump(template(name, sizing), sort_keys=False))
    (out / "sizing.json").write_text(json.dumps(sizing, indent=1))
    (out / "DEPLOY.md").write_text(STEPS.format(name=name, tag=tag, **sizing))
    return out
