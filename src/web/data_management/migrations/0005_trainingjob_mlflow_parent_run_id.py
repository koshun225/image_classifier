from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('data_management', '0004_auto_20251117_1540'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingjob',
            name='mlflow_parent_run_id',
            field=models.CharField(
                max_length=200,
                blank=True,
                null=True,
                verbose_name='MLflow 親ランID'
            ),
        ),
    ]

